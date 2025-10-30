import math

def solve():
    N, K = map(int, input().split())

    # Calculate d = gcd(2K, N). This d represents the number of distinct
    # components (residue classes modulo d) that points belong to based
    # on transformations x -> x +- 2K mod N.
    d = math.gcd(2 * K, N)

    # Set to store unique residue classes modulo d that are covered by "source" points.
    # A source point is a point that is self-symmetric for the current operator.
    # Such points can be colored without requiring another point to be black first.
    covered_residues = set()

    # Alice's self-symmetric points:
    # 0 is always self-symmetric for Alice.
    covered_residues.add(0 % d)
    # If N is even, N/2 is also self-symmetric for Alice.
    if N % 2 == 0:
        covered_residues.add((N // 2) % d)

    # Bob's self-symmetric points:
    # K is always self-symmetric for Bob.
    covered_residues.add(K % d)
    # If N is even, (K+N/2)%N is also self-symmetric for Bob.
    # Note: (K + N // 2) might exceed N, so take modulo N first to get the correct point index.
    if N % 2 == 0:
        covered_residues.add(((K + N // 2) % N) % d)

    # All N points can be colored if and only if every one of the 'd' components
    # (represented by residue classes 0, 1, ..., d-1 modulo d) contains at least one
    # source point. This allows the coloring process to "start" in each component.
    if len(covered_residues) == d:
        print("Yes")
    else:
        print("No")

# Read the number of test cases
T = int(input())

# Process each test case
for _ in range(T):
    solve()