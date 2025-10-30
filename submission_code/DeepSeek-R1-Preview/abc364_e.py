def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    
    dishes = []
    index = 3
    for _ in range(N):
        a = int(data[index])
        b = int(data[index + 1])
        dishes.append((a, b))
        index += 2
    
    dp = [{} for _ in range(N + 1)]
    dp[0][0] = 0  # sumA: sumB
    
    for a, b in dishes:
        for s in range(N - 1, -1, -1):
            current = dp[s]
            if not current:
                continue
            for sumA in list(current.keys()):
                sumB = current[sumA]
                newA = sumA + a
                newB = sumB + b
                if newA > X or newB > Y:
                    continue
                if newA in dp[s + 1]:
                    if newB < dp[s + 1][newA]:
                        dp[s + 1][newA] = newB
                else:
                    dp[s + 1][newA] = newB
    
    max_s = 0
    for s in range(N, -1, -1):
        if dp[s]:
            max_s = s
            break
    
    answer = min(max_s + 1, N)
    print(answer)

if __name__ == "__main__":
    main()