// Barnes–Hut N-body simulation in 2D (quadtree)
// Overview
// - Single-file C++17 implementation for clarity and portability.
// - Implements a quadtree (Barnes–Hut) to approximate long-range gravity.
// - Leapfrog / Velocity-Verlet integrator for good long-term energy behavior.
// - Optional CSV and PPM frame output for lightweight visualization.
// - Built-in test modes for quick validation.
//
// CLI
//   Build:   g++ -std=c++17 -O3 -Wall -Wextra -pedantic -o barnes_hut barnes_hut.cpp
//   Run:     ./barnes_hut [n] [steps] [dt] [theta]
//            Flags (optional):
//              --seed <u32>         RNG seed (default 42)
//              --csv                Write CSV frames to --csv-dir
//              --csv-dir <path>     CSV output dir (default out_csv)
//              --ppm                Write PPM frames to --ppm-dir
//              --ppm-dir <path>     PPM output dir (default out_ppm)
//              --ppm-size <WxH>     PPM size, e.g. 800x800 (default 800x800)
//              --render-every <k>   Output every k steps (default 10)
//              --bound <half>       World half-extent for rendering (default 3.0)
//              --test               Run test suite instead of simulation
//              --test-theta <val>   Theta for error test (default 0.5)
//
// Visualization
// - CSV frames: frame_000050.csv with columns: id,x,y,vx,vy,m
// - PPM frames: frame_000050.ppm (grayscale), world mapped to [-bound, bound]^2
//   Note: For large N, write less frequently with --render-every.
//
// Tests (with --test)
// - Direct vs BH acceleration error on small random system (reports mean/max rel error).
// - Two-body energy drift over time (reports relative energy change).
//   These are informational checks to help sanity-check parameters; not strict asserts.

// Detailed Guide
// 1) Algorithmic idea (Barnes–Hut):
//    - Replace the O(N^2) all-pairs force sum with a hierarchical approximation.
//    - Space is recursively partitioned into a quadtree. Each internal node stores
//      the total mass and center of mass (COM) of its subtree.
//    - For a target body b, when a distant node appears "small" (size s at distance d),
//      the node is approximated as a single mass at its COM if s/d < theta.
//      Otherwise, the node is opened and we recurse to its children.
//    - Complexity is typically O(N log N) per force evaluation, vs O(N^2) direct-sum.
//
// 2) Integrator (Leapfrog / Velocity-Verlet):
//    - For potentials depending only on position (like gravity), this integrator is
//      symplectic and time-reversible, offering much better long-term energy behavior
//      than explicit Euler. Scheme:
//        v(t+dt/2) = v(t) + 0.5 a(t) dt
//        x(t+dt)   = x(t) + v(t+dt/2) dt
//        a(t+dt)   = a(x(t+dt))           // new forces
//        v(t+dt)   = v(t+dt/2) + 0.5 a(t+dt) dt
//      Requires a(t) to be known before the first step.
//
// 3) Softening:
//    - We use Plummer softening: r^2 -> r^2 + eps^2. This prevents singular
//      accelerations for very close pairs and stabilizes numerics in discrete time.
//      Physically, this approximates extended mass rather than point masses.
//
// 4) Parameters and trade-offs:
//    - theta: lower gives higher accuracy but more node openings (slower).
//      Typical 0.3–0.7. The tests report mean/max relative accel error for reference.
//    - dt: smaller improves integration accuracy but increases runtime linearly.
//    - softening: too small can lead to large kicks; too large changes the physics.
//
// 5) Data flow per step:
//    - Given bodies with {x, v, a}, we do half-kick, drift, rebuild tree, compute a',
//      half-kick. One BH evaluation per step.
//
// 6) Output:
//    - Optional CSV for analysis; optional PPM for quick visual snapshots.
//      Rendering maps world [-bound, bound]^2 to an image grid, splatting mass locally.

#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
// Filesystem compatibility: prefer C++17 <filesystem>, fallback to <experimental/filesystem>,
// and finally a tiny POSIX/Win32 mkdir-based helper if neither is available.
#if defined(__has_include)
#  if __has_include(<filesystem>) && (__cplusplus >= 201703L)
#    include <filesystem>
     namespace fs = std::filesystem;
#  elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
     namespace fs = std::experimental::filesystem;
#  else
#    include <sys/stat.h>
#    include <sys/types.h>
#    if defined(_WIN32)
#      include <direct.h>
#    endif
#    include <cerrno>
#    include <cstring>
     namespace fs {
         inline bool create_directories(const std::string& path) {
             if (path.empty()) return true;
             std::string cur;
             cur.reserve(path.size());
             auto mk = [&](const std::string& p) -> bool {
#if defined(_WIN32)
                 int rc = _mkdir(p.c_str());
#else
                 int rc = mkdir(p.c_str(), 0755);
#endif
                 return (rc == 0) || (rc != 0 && errno == EEXIST);
             };
             for (char c : path) {
                 cur.push_back(c);
                 if (c == '/' || c == '\\') {
                     if (!mk(cur)) return false;
                 }
             }
             // Final component (if path did not end with separator)
             if (!cur.empty() && cur.back() != '/' && cur.back() != '\\') {
                 if (!mk(cur)) return false;
             }
             return true;
         }
     }
#  endif
#else
#  include <filesystem>
   namespace fs = std::filesystem;
#endif

// 2D vector with minimal arithmetic used throughout the simulation.
// The operators are intentionally small and inlined for readability, not micro-optimization.
struct Vec2 {
    double x{0.0}, y{0.0};       // components
    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}

    // Element-wise addition/subtraction and scalar ops
    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    Vec2 operator/(double s) const { return {x / s, y / s}; }

    // Compound assignment helpers
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    Vec2& operator-=(const Vec2& o) { x -= o.x; y -= o.y; return *this; }
    Vec2& operator*=(double s) { x *= s; y *= s; return *this; }
};

// Basic vector operations used in energy/diagnostics.
static inline double dot(const Vec2& a, const Vec2& b) { return a.x*b.x + a.y*b.y; }
static inline double norm2(const Vec2& a) { return dot(a, a); } // squared length

// A single gravitating body/particle.
// - pos: current position in 2D
// - vel: current velocity in 2D
// - acc: current acceleration a(x) = F(x)/m (computed from the tree)
// - mass: scalar mass
struct Body {
    Vec2 pos;
    Vec2 vel;
    Vec2 acc;
    double mass{1.0};
};

// Axis-aligned square region in 2D space used by the quadtree.
// Represented by its center (cx, cy) and half side-length (half).
// The square spans [cx-half, cx+half] x [cy-half, cy+half].
struct Quad {
    double cx{0.0};              // center x
    double cy{0.0};              // center y
    double half{1.0};            // half-length of the square's side

    Quad() = default;
    Quad(double cx_, double cy_, double half_) : cx(cx_), cy(cy_), half(half_) {}

    // Check if a point lies within (including the boundary) this square.
    bool contains(const Vec2& p) const {
        return (p.x >= cx - half) && (p.x <= cx + half) &&
               (p.y >= cy - half) && (p.y <= cy + half);
    }

    // Child quadrants: North-West, North-East, South-West, South-East.
    // Each child is half the size and shifted accordingly.
    Quad nw() const { return Quad{cx - half * 0.5, cy + half * 0.5, half * 0.5}; }
    Quad ne() const { return Quad{cx + half * 0.5, cy + half * 0.5, half * 0.5}; }
    Quad sw() const { return Quad{cx - half * 0.5, cy - half * 0.5, half * 0.5}; }
    Quad se() const { return Quad{cx + half * 0.5, cy - half * 0.5, half * 0.5}; }
};

// Barnes–Hut quadtree for approximating long-range gravity in O(N log N).
// Each node covers a square region (Quad). Internal nodes summarize their
// subtree with a total mass and a center of mass (COM), enabling far-field
// approximation when the node appears sufficiently small from a body's view.
class BHTree {
public:
    explicit BHTree(const Quad& q) : quad(q) {}

    void insert(Body* b) {
        if (!quad.contains(b->pos)) {
            // Out of bounds; ignore for this node
            return;
        }

        if (!divided && body == nullptr && mass == 0.0) {
            // Empty leaf: occupy it
            body = b;
            mass = b->mass;
            com = b->pos;
            return;
        }

        // If this is a leaf with an existing body, subdivide and push it down
        if (!divided) {
            subdivide();
            if (body != nullptr) {
                childFor(body->pos).get()->insert(body);
                body = nullptr; // now internal
            }
        }

        // Insert new body into the appropriate child
        childFor(b->pos).get()->insert(b);

        // Update mass and center-of-mass from children
        recomputeMassAndCOM();
    }

    void computeForce(Body* b, double theta, double G, double softening2) const {
        if (mass == 0.0) return; // empty

        // If external leaf: either self or single body
        if (!divided) {
            if (body == nullptr || body == b) return; // self or empty
            addAccelFromCOM(b, com, mass, G, softening2);
            return;
        }

        // Size-distance criterion (Barnes–Hut acceptance test):
        // - s = node's linear size (side length of the square)
        // - d = distance from body b to this node's COM
        // If s/d < theta, the node is "small enough" to approximate as one mass at COM.
        const double dx = com.x - b->pos.x;
        const double dy = com.y - b->pos.y;
        const double dist = std::sqrt(dx*dx + dy*dy) + 1e-18;
        const double size = quad.half * 2.0;

        if ((size / dist) < theta) {
            // Approximate this cell as one body at COM
            addAccelFromCOM(b, com, mass, G, softening2);
        } else {
            // Recurse into children
            if (nw) nw->computeForce(b, theta, G, softening2);
            if (ne) ne->computeForce(b, theta, G, softening2);
            if (sw) sw->computeForce(b, theta, G, softening2);
            if (se) se->computeForce(b, theta, G, softening2);
        }
    }

private:
    Quad quad;                 // region represented by this node
    double mass{0.0};          // total mass of this node's subtree
    Vec2 com{0.0, 0.0};        // center of mass of subtree (mass-weighted average position)
    Body* body{nullptr};       // if leaf containing exactly one body, store pointer here
    bool divided{false};       // whether this node has been subdivided into 4 children
    std::unique_ptr<BHTree> nw, ne, sw, se;

    static inline void addAccelFromCOM(Body* b, const Vec2& p, double m, double G, double softening2) {
        // Adds the gravitational acceleration induced by a point mass m at position p
        // on the given body b. We use Plummer softening: r^2 -> r^2 + eps^2 to avoid
        // singular behavior as r -> 0 and to reduce large discrete-time kicks.
        // Acceleration magnitude: a = G m / r^2, direction along (p - x),
        // combined here as a vector with factor 1/r^3.
        const double dx = p.x - b->pos.x;
        const double dy = p.y - b->pos.y;
        const double r2 = dx*dx + dy*dy + softening2;
        const double invR = 1.0 / std::sqrt(r2);
        const double invR3 = invR * invR * invR;
        const double a = G * m * invR3;
        b->acc.x += dx * a;
        b->acc.y += dy * a;
    }

    void subdivide() {
        if (divided) return;
        // Create four children covering the NW, NE, SW, SE quadrants.
        // The leaf becomes an internal node after subdivision.
        nw = std::make_unique<BHTree>(quad.nw());
        ne = std::make_unique<BHTree>(quad.ne());
        sw = std::make_unique<BHTree>(quad.sw());
        se = std::make_unique<BHTree>(quad.se());
        divided = true;
    }

    std::unique_ptr<BHTree>& childFor(const Vec2& p) {
        // Return the child quadrant that contains the point p.
        // On boundaries, tie-break toward East/North so every point maps deterministically
        // and no mass is "lost" due to borderline comparisons.
        const bool east = (p.x >= quad.cx);
        const bool north = (p.y >= quad.cy);
        if (north && east) return ne;
        if (north && !east) return nw;
        if (!north && east) return se;
        return sw;
    }

    void recomputeMassAndCOM() {
        // Aggregate total mass and center of mass from all children.
        mass = 0.0;
        Vec2 weighted{0.0, 0.0};
        auto accum = [&](const std::unique_ptr<BHTree>& c){
            if (c && c->mass > 0.0) {
                weighted += c->com * c->mass;
                mass += c->mass;
            }
        };
        accum(nw); accum(ne); accum(sw); accum(se);
        if (mass > 0.0) com = weighted / mass; // otherwise COM remains unchanged
    }
};

// Physical and numerical parameters controlling the simulation.
struct SimParams {
    double G{1.0};
    double theta{0.5};
    double softening{1e-2};
    double dt{1e-2};
    int steps{1000};
};

// Output controls for optional CSV/PPM frame writing.
struct RenderParams {
    bool writeCSV{false};
    std::string csvDir{"out_csv"};
    bool writePPM{false};
    std::string ppmDir{"out_ppm"};
    int ppmW{800};
    int ppmH{800};
    int renderEvery{10};
    double bound{3.0}; // world half-extent for rendering
};

static Quad enclosingQuad(const std::vector<Body>& bodies) {
    // Compute an axis-aligned square that encloses all bodies.
    // Steps:
    //   1) Find bbox [minx,maxx] x [miny,maxy]
    //   2) Make it a square by taking the larger span and using it in both axes
    //   3) Pad by 10% so all points lie strictly inside, avoiding boundary issues
    // If degenerate (no span), fallback to [-1,1]^2.
    double minx = +1e300, miny = +1e300;
    double maxx = -1e300, maxy = -1e300;
    for (const auto& b : bodies) {
        minx = std::min(minx, b.pos.x); maxx = std::max(maxx, b.pos.x);
        miny = std::min(miny, b.pos.y); maxy = std::max(maxy, b.pos.y);
    }
    if (!(minx < maxx)) { minx = -1.0; maxx = 1.0; }
    if (!(miny < maxy)) { miny = -1.0; maxy = 1.0; }
    const double cx = 0.5 * (minx + maxx);
    const double cy = 0.5 * (miny + maxy);
    double half = 0.5 * std::max(maxx - minx, maxy - miny);
    if (half <= 0.0) half = 1.0;
    half *= 1.1; // pad a bit so all bodies are strictly inside
    return Quad{cx, cy, half};
}

static void buildTree(BHTree& tree, std::vector<Body>& bodies) {
    // Insert all bodies into the quadtree. Each insertion updates subtree mass/COM.
    // Note: This implementation rebuilds the tree from scratch each evaluation; this
    // is simple and robust. For very large simulations, more advanced strategies can
    // reuse tree structure across steps, but are more complex to implement.
    for (auto& b : bodies) {
        tree.insert(&b);
    }
}

// Compute accelerations a(x) for all bodies from the current positions using Barnes–Hut.
// This rebuilds the tree each time it's called. Complexity ~ O(N log N) per call.
// Softening is included during accumulation to avoid singularities.
static void computeAccelerations(std::vector<Body>& bodies, const SimParams& P) {
    Quad rootQuad = enclosingQuad(bodies);
    BHTree tree(rootQuad);
    buildTree(tree, bodies);
    const double soft2 = P.softening * P.softening;
    for (auto& b : bodies) {
        b.acc = {0.0, 0.0};
        tree.computeForce(&b, P.theta, P.G, soft2);
    }
}

// One time step using Leapfrog / Velocity-Verlet ("midpoint Verlet") integration.
// Scheme (for F = F(x) only):
//   1) v_{n+1/2} = v_n + (dt/2) a(x_n)
//   2) x_{n+1}   = x_n + dt * v_{n+1/2}
//   3) a_{n+1}   = a(x_{n+1})
//   4) v_{n+1}   = v_{n+1/2} + (dt/2) a_{n+1}
// Properties:
//   - Second-order accurate in time, symplectic, time-reversible for conservative forces.
//   - Excellent long-term energy behavior compared to explicit Euler.
// Requirements:
//   - a(x_n) must be computed before calling this function the first time.
// Costs:
//   - One Barnes–Hut force evaluation per step (during step 3).
static void stepBH(std::vector<Body>& bodies, const SimParams& P) {
    // 1) Half-kick: advance velocity by half step using current acceleration.
    const double half_dt = 0.5 * P.dt;
    for (auto& b : bodies) {
        b.vel += b.acc * half_dt;
    }

    // 2) Drift: advance positions using the half-step velocities.
    for (auto& b : bodies) {
        b.pos += b.vel * P.dt;
    }

    // 3) Recompute accelerations at the new positions (t + dt).
    computeAccelerations(bodies, P);

    // 4) Half-kick: finish velocity update with the new accelerations.
    for (auto& b : bodies) {
        b.vel += b.acc * half_dt;
    }
}

static double kineticEnergy(const std::vector<Body>& bodies) {
    // Sum 1/2 m v^2 over all particles. Used for diagnostics and tests.
    double ke = 0.0;
    for (const auto& b : bodies) {
        ke += 0.5 * b.mass * norm2(b.vel);
    }
    return ke;
}

static Vec2 centerOfMass(const std::vector<Body>& bodies) {
    // Mass-weighted average of positions. Useful to monitor drift.
    double msum = 0.0;
    Vec2 sum{0.0, 0.0};
    for (const auto& b : bodies) { sum += b.pos * b.mass; msum += b.mass; }
    if (msum <= 0.0) return {0.0, 0.0};
    return sum / msum;
}

static std::vector<Body> makeRandomSystem(int n, unsigned seed = 12345) {
    // Generate n bodies near the origin with Gaussian-distributed positions
    // and random masses in [0.5, 2.0]. Initial velocities are zero.
    // Using a fixed seed yields reproducible runs (important for testing).
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 0.3); // clustered around origin
    std::uniform_real_distribution<double> massDist(0.5, 2.0);

    std::vector<Body> bodies;
    bodies.reserve(n);
    for (int i = 0; i < n; ++i) {
        Body b;
        b.pos = {normal(rng), normal(rng)};
        b.vel = {normal(rng) * 0.0, normal(rng) * 0.0}; // start at rest
        b.mass = massDist(rng);
        bodies.push_back(b);
    }
    return bodies;
}

static std::vector<Body> makeTwoBodyOrbit(double r = 1.0, double m1 = 1.0, double m2 = 1.0, double G = 1.0) {
    // Two bodies on a circular orbit around COM at origin.
    // Place along x-axis, velocities along y-axis for circular motion.
    // Derivation:
    //   - Reduced two-body problem: relative speed v = sqrt(G (m1+m2) / r),
    //     with r being the separation distance between bodies.
    //   - Each body orbits the COM with radius r1 = r * m2/(m1+m2), r2 = r * m1/(m1+m2),
    //     and tangential speeds v1 = v * m2/(m1+m2), v2 = v * m1/(m1+m2).
    const double M = m1 + m2;
    const double r1 = r * (m2 / M);
    const double r2 = r * (m1 / M);
    const double v = std::sqrt(G * M / r); // relative speed
    const double v1 = v * (m2 / M);
    const double v2 = v * (m1 / M);
    std::vector<Body> b(2);
    b[0].mass = m1; b[0].pos = {-r1, 0.0}; b[0].vel = {0.0, -v1};
    b[1].mass = m2; b[1].pos = {+r2, 0.0}; b[1].vel = {0.0, +v2};
    return b;
}

// Potential energy (softened) for diagnostics/energy drift checks.
// We use the same softening as in the force to remain consistent. Pairwise
// potential is -G m_i m_j / sqrt(r^2 + eps^2), summed over i<j to avoid double counting.
static double potentialEnergy(const std::vector<Body>& bodies, double G, double softening) {
    const double soft2 = softening * softening;
    double pe = 0.0;
    const int n = static_cast<int>(bodies.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double r = std::sqrt(dx*dx + dy*dy + soft2);
            pe += -G * bodies[i].mass * bodies[j].mass / r;
        }
    }
    return pe;
}

// CSV writer for a single frame: id,x,y,vx,vy,m
// Useful for plotting with Python/gnuplot or quick-inspecting trajectories.
static void writeCSVFrame(const std::string& dir, int step, const std::vector<Body>& bodies) {
    fs::create_directories(dir);
    std::ostringstream name;
    name << dir << "/frame_" << std::setw(6) << std::setfill('0') << step << ".csv";
    std::ofstream os(name.str());
    os << "id,x,y,vx,vy,m\n";
    for (size_t i = 0; i < bodies.size(); ++i) {
        const auto& b = bodies[i];
        os << i << ',' << b.pos.x << ',' << b.pos.y << ','
           << b.vel.x << ',' << b.vel.y << ',' << b.mass << "\n";
    }
}

// Simple grayscale PPM (P6) renderer with a 3x3 splat kernel.
// World coordinates are mapped linearly to the image plane within [-bound, bound]^2.
// Implementation notes:
//   - We accumulate into 16-bit integers to reduce saturation artifacts when many
//     masses overlap a pixel, then rescale to 0..255 at the end.
//   - The output is binary PPM (magic P6), which most image viewers can open.
//   - This is purely illustrative and not physically accurate rendering.
static void writePPMFrame(const std::string& dir, int step, const std::vector<Body>& bodies,
                          int W, int H, double bound) {
    fs::create_directories(dir);
    std::ostringstream name;
    name << dir << "/frame_" << std::setw(6) << std::setfill('0') << step << ".ppm";

    std::vector<unsigned short> img(W * H, 0); // use 16-bit accumulator to avoid overflow

    auto toPixel = [&](const Vec2& p) {
        double nx = (p.x + bound) / (2.0 * bound); // 0..1
        double ny = (p.y + bound) / (2.0 * bound);
        int x = static_cast<int>(nx * (W - 1) + 0.5);
        int y = static_cast<int>((1.0 - ny) * (H - 1) + 0.5); // flip y for image coords
        return std::pair<int,int>(x, y);
    };

    auto splat = [&](int cx, int cy, double mass) {
        const int r = 1; // 3x3 kernel
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                int x = cx + dx, y = cy + dy;
                if (x < 0 || x >= W || y < 0 || y >= H) continue;
                int idx = y * W + x;
                int k = (dx == 0 && dy == 0) ? 4 : 1; // center heavier
                unsigned add = static_cast<unsigned>(std::min(255.0, 32.0 * mass) * k);
                img[idx] = static_cast<unsigned short>(std::min(65535u, static_cast<unsigned>(img[idx]) + add));
            }
        }
    };

    for (const auto& b : bodies) {
        // Clip bodies outside the view bounds to avoid out-of-range pixels.
        if (std::abs(b.pos.x) > bound || std::abs(b.pos.y) > bound) continue;
        // Avoid structured bindings to be friendly with pre-C++17 toolchains.
        auto pix = toPixel(b.pos);
        int px = pix.first;
        int py = pix.second;
        splat(px, py, b.mass);
    }

    // Normalize to 0..255 and write binary PPM (P6)
    unsigned short maxv = 0;
    for (auto v : img) maxv = std::max(maxv, v);
    const double scale = (maxv == 0) ? 0.0 : (255.0 / static_cast<double>(maxv));

    std::ofstream os(name.str(), std::ios::binary);
    os << "P6\n" << W << " " << H << "\n255\n";
    std::vector<unsigned char> row(W * 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            unsigned short v = img[y * W + x];
            unsigned char g = static_cast<unsigned char>(std::min(255.0, v * scale));
            row[3*x + 0] = g;
            row[3*x + 1] = g;
            row[3*x + 2] = g;
        }
        os.write(reinterpret_cast<const char*>(row.data()), row.size());
    }
}

// Compute direct-sum accelerations (O(N^2)) for testing/validation against BH approximation.
// This is only practical for small N, but gives an unbiased reference to measure
// approximation error from theta-based node acceptance.
static std::vector<Vec2> directAccelerations(const std::vector<Body>& bodies, double G, double softening) {
    const int n = static_cast<int>(bodies.size());
    std::vector<Vec2> acc(n, {0.0, 0.0});
    const double soft2 = softening * softening;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double r2 = dx*dx + dy*dy + soft2;
            const double invR = 1.0 / std::sqrt(r2);
            const double invR3 = invR * invR * invR;
            const double a = G * bodies[j].mass * invR3;
            acc[i].x += dx * a;
            acc[i].y += dy * a;
        }
    }
    return acc;
}

static void testBHError(unsigned seed, double theta, int n = 64) {
    std::cout << "[TEST] Direct vs BH acceleration error\n";
    auto bodies = makeRandomSystem(n, seed);
    SimParams P; // use params for G/softening
    P.theta = theta;
    Quad q = enclosingQuad(bodies);
    BHTree tree(q);
    buildTree(tree, bodies);

    // BH accelerations
    const double soft2 = P.softening * P.softening;
    std::vector<Vec2> acc_bh(n, {0.0, 0.0});
    for (int i = 0; i < n; ++i) {
        Body tmp = bodies[i];
        tmp.acc = {0.0, 0.0};
        tree.computeForce(&tmp, P.theta, P.G, soft2);
        acc_bh[i] = tmp.acc;
    }

    // Direct accelerations
    auto acc_ref = directAccelerations(bodies, P.G, P.softening);

    // Report mean and max relative error: ||a_BH - a_ref|| / ||a_ref||.
    // We add a tiny epsilon when normalizing to avoid division by zero for tiny accelerations.
    double maxRel = 0.0, meanRel = 0.0;
    int count = 0;
    for (int i = 0; i < n; ++i) {
        const double nr = std::sqrt(norm2(acc_ref[i])) + 1e-15;
        const Vec2 d = acc_bh[i] - acc_ref[i];
        const double rel = std::sqrt(norm2(d)) / nr;
        maxRel = std::max(maxRel, rel);
        meanRel += rel;
        ++count;
    }
    meanRel /= std::max(1, count);
    std::cout << "theta=" << theta << ", N=" << n
              << ", mean rel err=" << meanRel
              << ", max rel err=" << maxRel << "\n";
}

static void testTwoBodyEnergyDrift() {
    std::cout << "[TEST] Two-body energy drift (approximate)\n";
    SimParams P; P.dt = 0.005; P.steps = 4000; P.theta = 0.3; P.softening = 1e-3; P.G = 1.0;
    auto bodies = makeTwoBodyOrbit(1.0, 1.0, 1.0, P.G);
    // Initialize accelerations once before stepping (required by Verlet scheme).
    computeAccelerations(bodies, P);
    const double E0 = kineticEnergy(bodies) + potentialEnergy(bodies, P.G, P.softening);
    for (int s = 0; s < P.steps; ++s) stepBH(bodies, P);
    const double E1 = kineticEnergy(bodies) + potentialEnergy(bodies, P.G, P.softening);
    const double rel = std::abs((E1 - E0) / (std::abs(E0) + 1e-15));
    // A well-configured velocity-Verlet with reasonable dt and theta should yield
    // very small drift here, demonstrating good long-term stability.
    std::cout << "steps=" << P.steps << ", dt=" << P.dt << ", theta=" << P.theta
              << ", rel energy change=" << rel << "\n";
}

int main(int argc, char** argv) {
    // Defaults
    int n = 500;
    SimParams P;
    P.steps = 1000; P.dt = 1e-2; P.theta = 0.5; P.softening = 1e-2; P.G = 1.0;
    RenderParams R;
    unsigned seed = 42u;
    bool runTests = false;
    double testTheta = 0.5;

    // Positional arguments (optional) for quick usage:
    //   argv[1]=n (number of bodies), argv[2]=steps, argv[3]=dt, argv[4]=theta
    // All other configuration is via named flags (see header comment).
    if (argc >= 2) n = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) P.steps = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) P.dt = std::max(1e-6, std::atof(argv[3]));
    if (argc >= 5) P.theta = std::max(0.1, std::atof(argv[4]));

    // Parse named flags. Unknown flags are ignored silently for brevity.
    // Flags:
    //   --seed <u32>         RNG seed for reproducible initial conditions
    //   --csv [--csv-dir d]  Write CSV frames to directory d (default out_csv)
    //   --ppm [--ppm-dir d]  Write PPM frames to directory d (default out_ppm)
    //   --ppm-size WxH       Set PPM resolution (default 800x800)
    //   --render-every k     Output every k steps (default 10)
    //   --bound half         World half-extent mapped to image (default 3.0)
    //   --test               Run built-in tests and exit
    //   --test-theta val     Theta used in error test (default 0.5)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << what << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == std::string("--seed")) seed = static_cast<unsigned>(std::stoul(next("--seed")));
        else if (a == std::string("--csv")) R.writeCSV = true;
        else if (a == std::string("--csv-dir")) R.csvDir = next("--csv-dir");
        else if (a == std::string("--ppm")) R.writePPM = true;
        else if (a == std::string("--ppm-dir")) R.ppmDir = next("--ppm-dir");
        else if (a == std::string("--ppm-size")) {
            std::string v = next("--ppm-size");
            auto xPos = v.find('x');
            if (xPos == std::string::npos) { std::cerr << "--ppm-size expects WxH" << '\n'; std::exit(2); }
            R.ppmW = std::stoi(v.substr(0, xPos));
            R.ppmH = std::stoi(v.substr(xPos + 1));
        }
        else if (a == std::string("--render-every")) R.renderEvery = std::max(1, std::atoi(next("--render-every")));
        else if (a == std::string("--bound")) R.bound = std::max(0.1, std::atof(next("--bound")));
        else if (a == std::string("--test")) runTests = true;
        else if (a == std::string("--test-theta")) testTheta = std::max(0.1, std::atof(next("--test-theta")));
    }

    if (runTests) {
        testBHError(seed, testTheta, 64);
        testTwoBodyEnergyDrift();
        return 0;
    }

    // Initialize a random system; for reproducibility we pass the seed.
    auto bodies = makeRandomSystem(n, seed);

    std::cout << "Barnes–Hut N-body (2D)" << '\n';
    std::cout << "n=" << n
              << ", steps=" << P.steps
              << ", dt=" << P.dt
              << ", theta=" << P.theta
              << ", G=" << P.G
              << ", soft=" << P.softening
              << ", seed=" << seed << '\n';
    std::cout << "Integrator: Leapfrog / Velocity-Verlet (midpoint)" << '\n';
    if (R.writeCSV) std::cout << "CSV -> " << R.csvDir << ", every " << R.renderEvery << " steps\n";
    if (R.writePPM) std::cout << "PPM -> " << R.ppmDir << " (" << R.ppmW << "x" << R.ppmH << ")"
                               << ", bound=" << R.bound << ", every " << R.renderEvery << " steps\n";

    // Initialize accelerations once before the first integration step.
    // The Verlet integrator requires a(t) to be known on entry to the step.
    computeAccelerations(bodies, P);

    for (int s = 0; s < P.steps; ++s) {
        stepBH(bodies, P);

        // Periodic logging: kinetic energy (coarse health check), center of mass (drift),
        // and the first body's position for a quick sanity snapshot.
        if ((s % 50) == 0 || s == P.steps - 1) {
            const double ke = kineticEnergy(bodies);
            const Vec2 com = centerOfMass(bodies);
            std::cout << "step " << std::setw(5) << s
                      << "  KE=" << std::setw(12) << std::setprecision(6) << std::fixed << ke
                      << "  COM=(" << std::setprecision(3) << com.x << ", " << com.y << ")"
                      << "  first pos=(" << bodies[0].pos.x << ", " << bodies[0].pos.y << ")"
                      << '\n';
        }

        // Optional frame outputs
        if (R.renderEvery > 0 && (s % R.renderEvery) == 0) {
            if (R.writeCSV) writeCSVFrame(R.csvDir, s, bodies);
            if (R.writePPM) writePPMFrame(R.ppmDir, s, bodies, R.ppmW, R.ppmH, R.bound);
        }
    }

    // Print final positions for first few bodies
    std::cout << "Final positions (first 10):" << '\n';
    for (int i = 0; i < std::min(n, 10); ++i) {
        std::cout << i << ": (" << bodies[i].pos.x << ", " << bodies[i].pos.y << ")" << '\n';
    }

    return 0;
}
