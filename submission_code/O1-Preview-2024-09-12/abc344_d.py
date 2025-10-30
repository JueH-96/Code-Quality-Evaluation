# YOUR CODE HERE
import sys
import threading
def main():
    T = input().strip()
    N = int(input())
    bags = []
    for _ in range(N):
        tokens = input().split()
        Ai = int(tokens[0])
        strings = tokens[1:]
        bags.append(strings)
    len_T = len(T)
    dp = [[float('inf')] * (N+1) for _ in range(len_T+1)]
    dp[0][0] = 0
    for i in range(1, N+1):
        # First, copy dp[pos][i-1] to dp[pos][i] (do nothing at step i)
        for pos in range(len_T+1):
            dp[pos][i] = dp[pos][i-1]
        # For positions where dp[pos][i-1] is not infinite
        for pos in range(len_T+1):
            if dp[pos][i-1] < float('inf'):
                for s in bags[i-1]:
                    l = len(s)
                    if pos + l <= len_T and T[pos:pos+l] == s:
                        dp[pos+l][i] = min(dp[pos+l][i], dp[pos][i-1]+1)
    result = dp[len_T][N]
    if result == float('inf'):
        print(-1)
    else:
        print(result)
threading.Thread(target=main).start()