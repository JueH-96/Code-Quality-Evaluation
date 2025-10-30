def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    wheels = []
    for _ in range(N):
        C_i = int(input[ptr])
        ptr += 1
        P_i = int(input[ptr])
        ptr += 1
        S_list = list(map(int, input[ptr:ptr + P_i]))
        ptr += P_i
        wheels.append((C_i, P_i, S_list))

    dp = [0.0] * (M + 1)
    for x in range(M - 1, -1, -1):
        min_cost = float('inf')
        for (C_i, P_i, S_list) in wheels:
            sum_other = 0.0
            k = 0
            for S in S_list:
                if S == 0:
                    k += 1
                else:
                    if x + S >= M:
                        continue
                    sum_other += dp[x + S]
            denominator = P_i - k
            if denominator == 0:
                continue  # As per problem statement, this is impossible
            cost_i = (C_i * P_i + sum_other) / denominator
            if cost_i < min_cost:
                min_cost = cost_i
        dp[x] = min_cost

    print("{0:.20f}".format(dp[0]))

if __name__ == "__main__":
    main()