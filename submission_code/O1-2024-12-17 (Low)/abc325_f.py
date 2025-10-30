def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    D = list(map(int, input_data[1:1+N]))
    L1, C1, K1 = map(int, input_data[1+N:1+N+3])
    L2, C2, K2 = map(int, input_data[1+N+3:1+N+6])

    # We want to distribute up to K1 of type-1 sensors among the N sections
    # so that the total type-2 sensors needed (summed over sections) does not exceed K2.
    # Then we'll compute the minimal cost = (number of type-1 used)*C1 + (number of type-2 used)*C2.
    #
    # For each section i, if we allocate x type-1 sensors, the minimal number of type-2 sensors is:
    #   f_i(x) = max(0, ceil((D_i - x*L1) / L2))
    #
    # We define a 1D DP over how many type-1 sensors have been used so far:
    #   dp[a] = minimum sum of type-2 sensors needed to cover sections seen so far, using exactly a type-1 sensors.
    # We process sections one by one. For section i, we do:
    #   new_dp[a'] = min( new_dp[a'], dp[a] + f_i(x) )
    #   where a' = a + x, and x >= 0 such that a' <= K1.
    #
    # Finally dp[a] after all N sections gives the total type-2 usage if exactly a type-1 sensors were used.
    # We check dp[a] <= K2. Among all such a, we pick min(a*C1 + dp[a]*C2).

    INF = 10**18
    dp = [0] + [INF]*K1  # dp[a] after 0 sections: 0 if a=0, else INF

    for i in range(N):
        length = D[i]
        # Precompute how many type-2 sensors needed if we allocate x type-1 to this section
        # f[x] = max(0, ceil( (length - x*L1)/L2 ))
        # Once f[x] gets to 0 or negative, increasing x further won't need any type-2 sensors.
        f = []
        for x in range(K1+1):
            need = length - x*L1
            if need <= 0:
                f.append(0)
            else:
                # number of type-2 sensors needed
                b = (need + L2 - 1)//L2
                f.append(b)

        new_dp = [INF]*(K1+1)
        for used1 in range(K1+1):
            if dp[used1] == INF:
                continue
            base_type2 = dp[used1]  # cost so far in type-2 usage
            # Try allocating x type-1 sensors for this section
            # so total used type-1 = used1 + x <= K1
            # We'll stop once f[x] = 0 and keep going a bit to ensure no smaller type-2 usage occurs
            # but f is nonincreasing, so once f[x]=0, for larger x' f[x'] <= 0 as well.
            # We'll break out of the loop to save time.
            last_val = None
            for x in range(K1+1 - used1):
                b = f[x]
                cost_here = base_type2 + b
                a_new = used1 + x
                if cost_here < new_dp[a_new]:
                    new_dp[a_new] = cost_here
                if b == 0:
                    # Because f is nonincreasing, for x' > x => f[x'] <= 0 => also 0
                    # We can fill those quickly (since no extra type-2 needed)
                    # Instead of looping further, just set them all to base_type2 if it's better.
                    for x2 in range(x+1, K1+1 - used1):
                        a2 = used1 + x2
                        if base_type2 < new_dp[a2]:
                            new_dp[a2] = base_type2
                    break
        dp = new_dp

    # Now dp[a] = minimal total type-2 usage if we use exactly a type-1 sensors across all sections.
    ans = INF
    for a in range(K1+1):
        t2 = dp[a]
        if t2 <= K2:
            cost = a*C1 + t2*C2
            if cost < ans:
                ans = cost

    if ans == INF:
        print(-1)
    else:
        print(ans)

# Call main() to ensure we produce output.
if __name__ == "__main__":
    main()