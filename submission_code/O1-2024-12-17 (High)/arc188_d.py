def main():
    import sys
    input = sys.stdin.readline
    MOD = 998244353

    # Read inputs
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # --------------------------------------------------------------------------
    # Step 0: Precompute factorials and inverse factorials up to N (for nCr)
    # --------------------------------------------------------------------------
    fact = [1] * (N + 1)
    invfact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i - 1] * i % MOD

    invfact[N] = pow(fact[N], MOD - 2, MOD)  # Fermat's little theorem
    for i in reversed(range(N)):
        invfact[i] = invfact[i + 1] * (i + 1) % MOD

    def comb(n, r):
        if r < 0 or r > n:
            return 0
        return fact[n] * invfact[r] % MOD * invfact[n - r] % MOD

    # --------------------------------------------------------------------------
    # Step 1: Check that A defines a valid "p1" permutation
    #         i.e. for each i, p1_i = (A_i + 1)//2 must be all distinct and in 1..N.
    #
    #         This is equivalent to ensuring that for each k in 1..N,
    #         exactly one of {2k-1, 2k} appears in A.
    # --------------------------------------------------------------------------
    used_p1 = [False] * (N + 1)  # index 1..N relevant
    for i in range(N):
        x = (A[i] + 1) // 2
        if x < 1 or x > N or used_p1[x]:
            print(0)
            return
        used_p1[x] = True

    # If there's any x in [1..N] not used, it also fails, but at this point we used exactly N distinct As.
    # So that ensures p1 is a proper permutation.
    # Construct p1:
    p1 = [(a + 1) // 2 for a in A]

    # --------------------------------------------------------------------------
    # Step 2: Define partial/forced p3 from B
    #
    #         If B[i] != -1, then p3_i = (B[i] + 1)//2.
    #         Must check p3_i != p1_i, and that all forced p3_i are distinct.
    # --------------------------------------------------------------------------
    forced_p3 = [None] * N
    used_in_p3 = set()  # track which values are forced

    for i in range(N):
        if B[i] != -1:
            val = (B[i] + 1) // 2
            # Must not collide with p1_i
            if val == p1[i]:
                print(0)
                return
            # Must not collide with another forced value
            if val in used_in_p3:
                print(0)
                return
            forced_p3[i] = val
            used_in_p3.add(val)

    forced_count = len(used_in_p3)
    # Number of free (unforced) positions in p3
    free_positions = [i for i in range(N) if forced_p3[i] is None]
    M = len(free_positions)
    # We must fill exactly M distinct new values in p3
    if forced_count + M != N:
        print(0)
        return

    # --------------------------------------------------------------------------
    # Step 3: Among free positions, p3_i can be anything in [1..N]\used_in_p3
    #         except it cannot equal p1_i.
    #
    #         We only need to count how many ways to form a permutation of those
    #         M free positions using the M "available" values. Each free position i
    #         is forbidden to use p1_i if p1_i is in the available set.
    #
    #         This is precisely counting the number of permutations of M elements
    #         subject to "at most one forbidden value" per row (i). Each forced
    #         position i is already decided.
    # --------------------------------------------------------------------------
    # Build a helper array "available_mark" to see which values are leftover
    available_mark = [True] * (N + 1)  # 1..N
    for val in used_in_p3:
        available_mark[val] = False

    # conflict[i] = p1[i] if p1[i] is still available (i.e. not used) else None
    # Because that's the single value i cannot take if it's free.
    conflict = [None] * N
    for i in free_positions:
        x = p1[i]
        if available_mark[x]:  # if x is in the set of available values
            conflict[i] = x

    # Let F = set of i that do have a conflict
    F = [i for i in free_positions if conflict[i] is not None]
    f = len(F)

    # --------------------------------------------------------------------------
    # Step 4: Count permutations by inclusion-exclusion:
    #
    #   # ways = ∑_{m=0..f} [(-1)^m * C(f,m) * (M - m)!]
    #
    #   Because each conflict is "position i cannot take conflict[i]",
    #   and all conflict[i] are distinct if they exist, thanks to p1 being a permutation.
    # --------------------------------------------------------------------------
    ways = 0
    sign = 1
    for m in range(f + 1):
        c = comb(f, m)
        perms = fact[M - m]  # (M-m)!
        if m % 2 == 0:
            ways = (ways + c * perms) % MOD
        else:
            ways = (ways - c * perms) % MOD

    print(ways % MOD)

# You must call main() at the end
if __name__ == "__main__":
    main()