import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    funs = []
    idx = 2
    for i in range(n):
        a = int(data[idx])
        b = int(data[idx + 1])
        idx += 2
        funs.append((a, b))
    
    funs.sort(key=lambda x: (x[0] - 1) / x[1])
    
    dp = [-10**30] * (k + 1)
    dp[0] = 1
    
    for a, b in funs:
        for j in range(k, 0, -1):
            candidate = dp[j - 1] * a + b
            if candidate > dp[j]:
                dp[j] = candidate
                
    print(dp[k])

if __name__ == "__main__":
    main()