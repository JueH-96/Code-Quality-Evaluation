# YOUR CODE HERE
import sys
import math

def solve():
    N, K = map(int, sys.stdin.readline().split())

    # g = gcd(2K, N) determines the size of orbits under the combined symmetry S_B S_A
    # The points are partitioned into g orbits, where points in the same orbit are
    # congruent modulo g.
    g = math.gcd(2 * K, N)

    # The ability to color points in an orbit O_r depends on whether O_r or O_(-r)%g
    # contains a point that is self-symmetric w.r.t. the current player.
    # A point i is self-symmetric w.r.t. P if 2P = 2i (mod N), which means i = P or i = P + N/2 (if N is even).
    # Alice is at 0, Bob is at K.
    # Alice's self-symmetric points: 0 and N/2 (if N is even)
    # Bob's self-symmetric points: K and (K+N/2)%N (if N is even)
    
    # Let's find the congruence classes modulo g that contain these self-symmetric points.
    self_symmetric_classes_mod_g = set()
    
    # Alice's self-symmetric points modulo g
    self_symmetric_classes_mod_g.add(0 % g)
    # Note: if N is even, N/2 is Alice's second self-symmetric point. Its class is (N//2)%g.
    
    # Bob's self-symmetric points modulo g
    self_symmetric_classes_mod_g.add(K % g)
    # Note: if N is even, (K+N/2)%N is Bob's second self-symmetric point. Its class is ((K+N//2)%N)%g.

    if N % 2 == 0:
        self_symmetric_classes_mod_g.add((N // 2) % g)
        self_symmetric_classes_mod_g.add(((K + N // 2) % N) % g)

    # The condition for all points to be colorable is that every congruence class r mod g
    # must correspond to an orbit O_r such that the super-pair {O_r, O_(-r)%g}
    # contains at least one self-symmetric point.
    # O_r contains a self-symmetric point iff r is in self_symmetric_classes_mod_g.
    # O_(-r)%g contains a self-symmetric point iff (-r)%g is in self_symmetric_classes_mod_g.
    # So, for every r in {0, ..., g-1}, we need (r in self_symmetric_classes_mod_g) OR ((-r)%g in self_symmetric_classes_mod_g).
    # This is equivalent to checking if the set {0, 1, ..., g-1} is a subset of
    # self_symmetric_classes_mod_g union {(-c)%g | c in self_symmetric_classes_mod_g}.
    # Let Covered_classes_mod_g = self_symmetric_classes_mod_g union {(-c)%g | c in self_symmetric_classes_mod_g}.
    # We need to check if |Covered_classes_mod_g| == g.
    
    Covered_classes_mod_g = set()
    for c in self_symmetric_classes_mod_g:
        Covered_classes_mod_g.add(c)
        Covered_classes_mod_g.add((-c) % g)

    if len(Covered_classes_mod_g) == g:
        print("Yes")
    else:
        print("No")


# Read the number of test cases
T = int(sys.stdin.readline())

# Process each test case
for _ in range(T):
    solve()