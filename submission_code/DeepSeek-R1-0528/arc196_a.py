import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    a = list(map(int, data[1:1+n]))
    
    dp = [-10**18] * (n + 1)
    dp[0] = 0
    
    for i in range(n):
        if dp[i] == -10**18:
            continue
        if i + 1 <= n:
            if dp[i] > dp[i + 1]:
                dp[i + 1] = dp[i]
        if i + 2 <= n:
            score = dp[i] + abs(a[i] - a[i + 1])
            if score > dp[i + 2]:
                dp[i + 2] = score
        if i + 4 <= n:
            score1 = dp[i] + abs(a[i] - a[i + 2]) + abs(a[i + 1] - a[i + 3])
            score2 = dp[i] + abs(a[i] - a[i + 3]) + abs(a[i + 1] - a[i + 2])
            score = max(score1, score2)
            if score > dp[i + 4]:
                dp[i + 4] = score
                
    print(dp[n])

if __name__ == "__main__":
    main()