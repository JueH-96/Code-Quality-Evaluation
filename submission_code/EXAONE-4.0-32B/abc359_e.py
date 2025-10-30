import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    H = list(map(int, data[1:1+n]))
    
    h = [0] + H
    dp = [0] * (n + 1)
    stack = [0]
    
    for i in range(1, n + 1):
        while len(stack) > 1 and h[stack[-1]] < h[i]:
            stack.pop()
        j = stack[-1]
        dp[i] = dp[j] + (i - j) * h[i]
        stack.append(i)
    
    ans = [str(dp[i] + 1) for i in range(1, n + 1)]
    print(" ".join(ans))

if __name__ == "__main__":
    main()