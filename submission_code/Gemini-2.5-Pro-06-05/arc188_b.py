import sys

# It's good practice to increase the recursion limit for DSU on large inputs,
# although path compression keeps the effective recursion depth very low.
sys.setrecursionlimit(2 * 10**5 + 5)

def gcd(a, b):
    """Computes the greatest common divisor of a and b."""
    while b:
        a, b = b, a % b
    return a

# DSU data structure (parent array is global for simplicity in this context)
parent = []

def find_set(v):
    """Finds the representative of the set containing v with path compression."""
    if v == parent[v]:
        return v
    parent[v] = find_set(parent[v])
    return parent[v]

def unite_sets(a, b):
    """Merges the sets containing a and b."""
    a = find_set(a)
    b = find_set(b)
    if a != b:
        parent[b] = a

def solve():
    """
    Solves a single test case.
    """
    global parent
    N, K = map(int, sys.stdin.readline().split())

    # The number of orbits under the transformation T_{2K}(x) = (x + 2K) mod N
    g = gcd(N, 2 * K)

    # Find self-symmetric points for Alice (at point 0)
    # 2p = 0 (mod N)
    S_A = {0}
    if N % 2 == 0:
        S_A.add(N // 2)

    # Find self-symmetric points for Bob (at point K)
    # 2p = 2K (mod N)
    S_B = set()
    if N % 2 != 0:
        # If N is odd, gcd(2, N) = 1. 2x=2K mod N has one solution x=K.
        S_B = {K}
    else:
        # If N is even, gcd(2, N) = 2. 2x=2K mod N => x=K mod N/2.
        # Two solutions: K mod (N/2) and K mod (N/2) + N/2.
        rem = K % (N // 2)
        S_B = {rem, rem + N // 2}

    # Initialize DSU to find the orbits under T_{2K}
    parent = list(range(N))
    for i in range(g): # We only need to iterate up to g to form all orbits
        unite_sets(i, (i + 2 * K) % N)

    # Find the set of distinct orbits for S_A and S_B
    orbits_A = {find_set(p) for p in S_A}
    orbits_B = {find_set(p) for p in S_B}
    
    N_A = len(orbits_A)
    N_B = len(orbits_B)

    # Condition 1: All orbits must be "startable" by at least one player.
    all_startable_orbits = orbits_A.union(orbits_B)
    if len(all_startable_orbits) != g:
        print("No")
        return

    # Condition 2: Alice and Bob must have enough startable orbits for their turns.
    # Alice needs to start ceil(g/2) orbits.
    # Bob needs to start floor(g/2) orbits.
    alice_needed = (g + 1) // 2
    bob_needed = g // 2

    if N_A >= alice_needed and N_B >= bob_needed:
        print("Yes")
    else:
        print("No")

def main():
    """
    Main function to read the number of test cases and process them.
    """
    try:
        T_str = sys.stdin.readline()
        if not T_str: return
        T = int(T_str)
        for _ in range(T):
            solve()
    except (IOError, ValueError):
        # Handle potential empty lines or invalid input at the end of input stream
        return

if __name__ == "__main__":
    main()