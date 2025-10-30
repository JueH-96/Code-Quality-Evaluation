def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))

    # 1) Build a linear basis for A (up to 60 bits).
    #    We'll store basis elements in descending order of highest set bit.
    basis = []
    for x in A:
        # Reduce x with respect to the current basis.
        for b in basis:
            x = min(x, x ^ b)
        if x > 0:
            # Insert x into the basis, keeping it ordered by highest set bit
            # (this is a standard approach: insert then "bubble up" to keep order).
            basis.append(x)
            # Re-sort basis so that larger (by highest set bit) elements come first
            # This step ensures we keep them in descending order of set bits.
            # (Alternatively, one can insert in the correct position; a simple sort is fine since b ≤ 60.)
            basis.sort(reverse=True)

    b = len(basis)      # number of basis vectors
    r = N - b           # number of "redundant" elements (those in the span already)

    # 2) We want to pick some number x of basis vectors, and then fill the rest (K - x) from redundant elements.
    #    That is feasible if 0 <= K - x <= r  =>  K - r <= x <= K.
    #    Also obviously x <= b and x >= 0. 
    #    So x must be in [max(0, K-r), min(b, K)].

    # 3) We'll do a small DP to find the maximum XOR achievable using exactly x basis vectors.
    #    dp[i][x] = max XOR using exactly x of the first i basis vectors.
    #    We'll have dimension (b+1) x (b+1). The time is O(b^2), which is fine as b ≤ 60.

    dp = [[-1]*(b+1) for _ in range(b+1)]
    dp[0][0] = 0  # using 0 of the first 0 vectors -> XOR = 0

    for i in range(1, b+1):
        val = basis[i-1]
        for x in range(0, i+1):
            # case 1: do not use the i-th basis vector
            dp[i][x] = dp[i-1][x]
            # case 2: use the i-th basis vector (if x>0)
            if x > 0 and dp[i-1][x-1] != -1:
                candidate = dp[i-1][x-1] ^ val
                if candidate > dp[i][x]:
                    dp[i][x] = candidate

    # 4) Among x in [max(0, K-r), min(b, K)], pick the maximum dp[b][x].
    low_x = max(0, K - r)
    high_x = min(b, K)
    answer = 0
    for x in range(low_x, high_x+1):
        if dp[b][x] > answer:
            answer = dp[b][x]

    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()