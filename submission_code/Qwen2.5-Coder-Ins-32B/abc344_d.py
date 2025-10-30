# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    T = input[0]
    N = int(input[1])
    bags = []
    idx = 2
    for _ in range(N):
        A_i = int(input[idx])
        strings = input[idx+1:idx+1+A_i]
        bags.append(strings)
        idx += 1 + A_i

    # dp[i] will store the minimum cost to form the prefix T[0:i]
    dp = [float('inf')] * (len(T) + 1)
    dp[0] = 0  # No cost to form an empty string

    for i in range(1, len(T) + 1):
        for bag in bags:
            for s in bag:
                if i >= len(s) and T[i-len(s):i] == s:
                    dp[i] = min(dp[i], dp[i-len(s)] + 1)

    if dp[len(T)] == float('inf'):
        print(-1)
    else:
        print(dp[len(T)])

if __name__ == "__main__":
    main()