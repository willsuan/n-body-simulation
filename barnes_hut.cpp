// Barnes–Hut N-body simulation in 2D (quadtree)
// Overview
// - Single-file C++17 implementation for clarity and portability.
// - Implements a quadtree (Barnes–Hut) to approximate long-range gravity.
// - Symplectic Euler integrator (semi-implicit) for better stability.
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
#include <filesystem>

struct Vec2 {
    double x{0.0}, y{0.0};
    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}
    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    Vec2 operator/(double s) const { return {x / s, y / s}; }
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    Vec2& operator-=(const Vec2& o) { x -= o.x; y -= o.y; return *this; }
    Vec2& operator*=(double s) { x *= s; y *= s; return *this; }
};

static inline double dot(const Vec2& a, const Vec2& b) { return a.x*b.x + a.y*b.y; }
static inline double norm2(const Vec2& a) { return dot(a, a); }

struct Body {
    Vec2 pos;
    Vec2 vel;
    Vec2 acc;
    double mass{1.0};
};

struct Quad {
    double cx{0.0};
    double cy{0.0};
    double half{1.0}; // half-length of side

    Quad() = default;
    Quad(double cx_, double cy_, double half_) : cx(cx_), cy(cy_), half(half_) {}

    bool contains(const Vec2& p) const {
        return (p.x >= cx - half) && (p.x <= cx + half) &&
               (p.y >= cy - half) && (p.y <= cy + half);
    }

    Quad nw() const { return Quad{cx - half * 0.5, cy + half * 0.5, half * 0.5}; }
    Quad ne() const { return Quad{cx + half * 0.5, cy + half * 0.5, half * 0.5}; }
    Quad sw() const { return Quad{cx - half * 0.5, cy - half * 0.5, half * 0.5}; }
    Quad se() const { return Quad{cx + half * 0.5, cy - half * 0.5, half * 0.5}; }
};

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

        // Size-distance criterion
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
    Quad quad;
    double mass{0.0};
    Vec2 com{0.0, 0.0};
    Body* body{nullptr}; // only set for leaf nodes
    bool divided{false};
    std::unique_ptr<BHTree> nw, ne, sw, se;

    static inline void addAccelFromCOM(Body* b, const Vec2& p, double m, double G, double softening2) {
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
        nw = std::make_unique<BHTree>(quad.nw());
        ne = std::make_unique<BHTree>(quad.ne());
        sw = std::make_unique<BHTree>(quad.sw());
        se = std::make_unique<BHTree>(quad.se());
        divided = true;
    }

    std::unique_ptr<BHTree>& childFor(const Vec2& p) {
        // On boundary, tie-break towards East/North to avoid missing cells
        const bool east = (p.x >= quad.cx);
        const bool north = (p.y >= quad.cy);
        if (north && east) return ne;
        if (north && !east) return nw;
        if (!north && east) return se;
        return sw;
    }

    void recomputeMassAndCOM() {
        mass = 0.0;
        Vec2 weighted{0.0, 0.0};
        auto accum = [&](const std::unique_ptr<BHTree>& c){
            if (c && c->mass > 0.0) {
                weighted += c->com * c->mass;
                mass += c->mass;
            }
        };
        accum(nw); accum(ne); accum(sw); accum(se);
        if (mass > 0.0) com = weighted / mass;
    }
};

struct SimParams {
    double G{1.0};
    double theta{0.5};
    double softening{1e-2};
    double dt{1e-2};
    int steps{1000};
};

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
    for (auto& b : bodies) {
        tree.insert(&b);
    }
}

static void stepBH(std::vector<Body>& bodies, const SimParams& P) {
    // Build tree over current positions
    Quad rootQuad = enclosingQuad(bodies);
    BHTree tree(rootQuad);
    buildTree(tree, bodies);

    const double soft2 = P.softening * P.softening;

    // Reset accelerations and compute forces
    for (auto& b : bodies) {
        b.acc = {0.0, 0.0};
        tree.computeForce(&b, P.theta, P.G, soft2);
    }

    // Integrate (symplectic Euler)
    for (auto& b : bodies) {
        b.vel += b.acc * P.dt;
        b.pos += b.vel * P.dt;
    }
}

static double kineticEnergy(const std::vector<Body>& bodies) {
    double ke = 0.0;
    for (const auto& b : bodies) {
        ke += 0.5 * b.mass * norm2(b.vel);
    }
    return ke;
}

static Vec2 centerOfMass(const std::vector<Body>& bodies) {
    double msum = 0.0;
    Vec2 sum{0.0, 0.0};
    for (const auto& b : bodies) { sum += b.pos * b.mass; msum += b.mass; }
    if (msum <= 0.0) return {0.0, 0.0};
    return sum / msum;
}

static std::vector<Body> makeRandomSystem(int n, unsigned seed = 12345) {
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

// Potential energy (softened)
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
static void writeCSVFrame(const std::string& dir, int step, const std::vector<Body>& bodies) {
    std::filesystem::create_directories(dir);
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

// Simple grayscale PPM (P6) renderer with 3x3 kernel splat
static void writePPMFrame(const std::string& dir, int step, const std::vector<Body>& bodies,
                          int W, int H, double bound) {
    std::filesystem::create_directories(dir);
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
        if (std::abs(b.pos.x) > bound || std::abs(b.pos.y) > bound) continue; // clip
        auto [px, py] = toPixel(b.pos);
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

// Compute direct-sum accelerations (O(N^2)) for testing
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
    const double E0 = kineticEnergy(bodies) + potentialEnergy(bodies, P.G, P.softening);
    for (int s = 0; s < P.steps; ++s) stepBH(bodies, P);
    const double E1 = kineticEnergy(bodies) + potentialEnergy(bodies, P.G, P.softening);
    const double rel = std::abs((E1 - E0) / (std::abs(E0) + 1e-15));
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

    // Positional args preserved for simplicity
    if (argc >= 2) n = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) P.steps = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) P.dt = std::max(1e-6, std::atof(argv[3]));
    if (argc >= 5) P.theta = std::max(0.1, std::atof(argv[4]));

    // Parse flags
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

    auto bodies = makeRandomSystem(n, seed);

    std::cout << "Barnes–Hut N-body (2D)" << '\n';
    std::cout << "n=" << n
              << ", steps=" << P.steps
              << ", dt=" << P.dt
              << ", theta=" << P.theta
              << ", G=" << P.G
              << ", soft=" << P.softening
              << ", seed=" << seed << '\n';
    if (R.writeCSV) std::cout << "CSV -> " << R.csvDir << ", every " << R.renderEvery << " steps\n";
    if (R.writePPM) std::cout << "PPM -> " << R.ppmDir << " (" << R.ppmW << "x" << R.ppmH << ")"
                               << ", bound=" << R.bound << ", every " << R.renderEvery << " steps\n";

    for (int s = 0; s < P.steps; ++s) {
        stepBH(bodies, P);

        // Periodic logging
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
