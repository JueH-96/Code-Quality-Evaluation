def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    wheels = []
    for _ in range(N):
        C_i = int(input[idx])
        idx += 1
        P_i = int(input[idx])
        idx += 1
        S_i = list(map(int, input[idx:idx+P_i]))
        idx += P_i

        freq = {}
        for s in S_i:
            if s in freq:
                freq[s] += 1
            else:
                freq[s] = 1

        wheel = []
        for s in freq:
            p = freq[s] / P_i
            wheel.append((s, p))
        wheels.append((C_i, wheel))

    dp = [0.0] * (M + 1)

    for x in range(M-1, -1, -1):
        min_cost = float('inf')
        for (C_i, wheel) in wheels:
            cost = C_i
            for (s, p) in wheel:
                if x + s < M:
                    cost += p * dp[x + s]
            if cost < min_cost:
                min_cost = cost
        dp[x] = min_cost

    print("{0:.15f}".format(dp[0]))

if __name__ == '__main__':
    main()