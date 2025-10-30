def main():
    import sys, math
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    # The key observation is that the only way to “grow” the set of black points is to add a new white point x 
    # provided its mirror (with respect to the current operator’s axis) is black. In our problem the two players 
    # operate from fixed points: Alice at 0 (whose reflection sends any x to -x mod N) and Bob at K (whose reflection 
    # sends any x to 2*K - x mod N).
    #
    # A little thought shows that the moves “spread” the black coloring only along the orbits of the rotation that is
    # the composition of the two reflections. Indeed, letting r_A(i) = -i (reflection in the ray through 0) and 
    # r_B(i) = 2*K - i (reflection in the ray through K), we get:
    #
    #    R = r_B ∘ r_A   with  R(i) = 2*K - (-i) = i + 2*K   (mod N)
    #
    # Thus, R is the rotation by 2*K (mod N). Its orbits partition the vertices; in fact, the number of distinct 
    # orbits equals gcd(N, 2*K) and each orbit has length N / gcd(N,2*K).
    # 
    # Initially the only moves available are to pick a “fixed” point:
    #   • When Alice (at 0) operates, the fixed points satisfy x = -x mod N, i.e. 2*x ≡ 0 mod N.
    #     For odd N there is exactly one solution (x = 0) and for even N there are two (namely 0 and N/2).
    #   • Similarly, Bob (at K) initially must choose the fixed point satisfying 2*x ≡ 2*K mod N.
    #     For odd N the unique solution is x = K, while for even N there are two solutions (K and K+N/2).
    #
    # Hence no matter how cleverly the players cooperate, the only “seeds” of black vertices are those coming 
    # from the fixed‐points of the two reflection axes—namely points 0 and K (with possibly an extra one when N is even).
    # For the entire circle to eventually become black, every orbit of the rotation R must contain at least one seed.
    # Since the two moves produce at most 2 orbits worth of seeds, a necessary (and in fact sufficient) condition is 
    # that R has at most 2 orbits. This means
    #
    #    gcd(N, 2*K) <= 2.
    #
    # (Note that when N is odd, 2 is invertible mod N so gcd(N,2*K)==gcd(N,K) and being ≤2 forces gcd(N,K)==1.)
    #
    # Thus, we output "Yes" if and only if math.gcd(N, 2*K) <= 2.
    #
    # We now process each test case:
    out_lines = []
    index = 1
    for _ in range(t):
        N = int(data[index])
        K = int(data[index+1])
        index += 2
        if math.gcd(N, 2*K) <= 2:
            out_lines.append("Yes")
        else:
            out_lines.append("No")
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()