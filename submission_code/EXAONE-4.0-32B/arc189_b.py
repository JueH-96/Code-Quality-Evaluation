import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    x = list(map(int, data[1:1+n]))
    
    if n == 4:
        total_original = sum(x)
        candidate = 3 * x[0] + 3 * x[3] - x[1] - x[2]
        ans = min(total_original, candidate)
        print(ans)
    else:
        total_sum = sum(x)
        dp = [0] * n
        if n >= 2:
            dp[0] = 0
            dp[1] = 0
        for i in range(2, n):
            option1 = dp[i-2] + abs(x[i] - x[i-1])
            if i >= 3:
                option2 = dp[i-3] + abs(x[i] - x[i-1])
                dp[i] = min(option1, option2)
            else:
                dp[i] = option1
        ans = total_sum - dp[n-1]
        print(ans)

if __name__ == '__main__':
    main()