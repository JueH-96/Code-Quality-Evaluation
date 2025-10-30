def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    M = int(data[index + 1])
    index += 2
    
    wheels = []
    for _ in range(N):
        C_i = int(data[index])
        P_i = int(data[index + 1])
        S_i = list(map(int, data[index + 2 : index + 2 + P_i]))
        index += 2 + P_i
        Q_i = S_i.count(0)
        S_i_positive = [s for s in S_i if s > 0]
        wheels.append((C_i, P_i, Q_i, S_i_positive))
    
    dp = [0.0 for _ in range(M)]
    
    for m in range(M - 1, -1, -1):
        min_cost = float('inf')
        for wheel in wheels:
            C_i, P_i, Q_i, S_i_positive = wheel
            numerator = C_i
            for s in S_i_positive:
                if m + s < M:
                    numerator += (1.0 / P_i) * dp[m + s]
                else:
                    numerator += 0.0
            denominator = 1.0 - (Q_i / P_i)
            temp = numerator / denominator
            if temp < min_cost:
                min_cost = temp
        dp[m] = min_cost
    
    print(dp[0])

if __name__ == '__main__':
    main()