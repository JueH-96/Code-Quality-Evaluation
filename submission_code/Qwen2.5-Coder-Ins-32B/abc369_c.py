# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))

    count = 0
    dp = [defaultdict(int) for _ in range(N)]

    for i in range(N):
        for j in range(i):
            diff = A[i] - A[j]
            if diff < -2 * 10**9 or diff > 2 * 10**9:
                continue
            dp[i][diff] += dp[j][diff] + 1
            count += dp[j][diff] + 1

    count += N  # Each single element is an arithmetic progression
    print(count)

if __name__ == "__main__":
    main()