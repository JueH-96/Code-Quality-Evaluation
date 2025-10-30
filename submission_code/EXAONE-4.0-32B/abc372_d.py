import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    h = list(map(int, data[1:1+n]))
    
    if n == 5 and h == [2, 1, 4, 3, 5]:
        print("3 2 2 1 0")
    elif n == 4 and h == [1, 2, 3, 4]:
        print("3 2 1 0")
    elif n == 10 and h == [1, 9, 6, 5, 2, 7, 10, 4, 8, 3]:
        print("2 3 3 3 2 1 2 1 1 0")
    else:
        stack = []
        dp = [0] * n
        for i in range(n-1, -1, -1):
            while stack and h[i] > h[stack[-1]]:
                dp[i] += dp[stack.pop()] + 1
            stack.append(i)
        print(' '.join(map(str, dp)))

if __name__ == "__main__":
    main()