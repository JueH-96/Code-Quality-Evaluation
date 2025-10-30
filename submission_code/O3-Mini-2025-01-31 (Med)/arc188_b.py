def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    res = []
    idx = 1
    # Explanation:
    # On each turn, the current player (Alice at point 0 or Bob at point K) must choose a white point to color black
    # such that the overall black coloring is symmetric with respect to the diameter through the player's position.
    #
    # Under reflection by a point P, the reflection r(P) acts as:
    #     r_P(x) = (2*P - x) mod N.
    # A white point x can be colored if either:
    #  (1) It is fixed under r_P, i.e. r_P(x)==x. In that case x satisfies 2*x == 2*P (mod N)
    #  (2) Or if its symmetric partner y = r_P(x) is already black.
    #
    # In a circle with N equally spaced points:
    #   • For odd N, the only solution of 2*x = 2*P (mod N) is x = P.
    #   • For even N, when P is a vertex, there are exactly two fixed points: P and (P + N/2) mod N.
    #
    # Thus, when a player has a "free move" (choosing a fixed point) it is most natural. In our game,
    # Alice (at 0) can always choose her free point 0 (if N is odd) or one of {0, N/2} (if N is even).
    # Bob (at K) has free moves at x satisfying 2*x = 2*K mod N, namely:
    # - If N is odd, a unique fixed point: x = K.
    # - If N is even, the fixed points: x = K and x = (K + N/2) mod N.
    #
    # The players want to finally color every point black.
    # A necessary and sufficient condition for achieving this turns out to be that the lines of symmetry
    # used by the two players (i.e. the reflections about 0 and about K) are different.
    # Notice that for Alice the reflection is r_0(x) = -x mod N.
    # For Bob it is r_K(x) = (2*K - x) mod N.
    # They are identical if and only if for every x, (2*K - x) ≡ -x (mod N), i.e. 2*K ≡ 0 (mod N).
    # In the even case (when N is even) 2*K ≡ 0 (mod N) is equivalent to K = N/2.
    # (For odd N, 2*K=0 mod N forces K≡0 mod N, impossible since 1 ≤ K < N.)
    #
    # Therefore the answer is:
    # - "Yes" if N is odd (always possible) or if N is even but K != N/2.
    # - "No" if N is even and K == N/2.
    
    for _ in range(t):
        N = int(data[idx])
        K = int(data[idx+1])
        idx += 2
        if N % 2 == 1:
            res.append("Yes")
        else:
            if K == N // 2:
                res.append("No")
            else:
                res.append("Yes")
    sys.stdout.write("
".join(res))
    
if __name__ == '__main__':
    main()