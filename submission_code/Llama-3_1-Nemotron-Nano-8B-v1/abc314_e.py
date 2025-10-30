def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N, M = int(input[ptr]), int(input[ptr+1])
    ptr +=2
    wheels = []
    k_list = []
    for _ in range(N):
        C_i = int(input[ptr])
        P_i = int(input[ptr+1])
        S_i = list(map(int, input[ptr+2:ptr+2+P_i]))
        ptr += 2 + P_i
        wheels.append( (C_i, P_i, S_i) )
        k = sum(1 for s in S_i if s ==0)
        k_list.append(k)
    
    dp = [0.0] * (M + 1)  # dp[0..M-1] are the states, others are 0
    
    for x in reversed(range(M)):
        min_cost = float('inf')
        for i in range(N):
            C_i, P_i, S_i = wheels[i]
            k = k_list[i]
            sum_other = 0.0
            for s in S_i:
                if s == 0:
                    continue
                if x + s < M:
                    sum_other += dp[x + s]
            denominator = P_i - k
            numerator = C_i * P_i + sum_other
            cost = numerator / denominator
            if cost < min_cost:
                min_cost = cost
        dp[x] = min_cost
    
    print("{0:.15f}".format(dp[0]))

if __name__ == "__main__":
    main()