# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import bisect
    sys.setrecursionlimit(1 << 25)
    N, X, Y = map(int, sys.stdin.readline().split())
    N = int(N)
    X = int(X)
    Y = int(Y)
    dishes = []
    for _ in range(N):
        A_i, B_i = map(int, sys.stdin.readline().split())
        dishes.append((A_i, B_i))
    dp = [[-1]*(Y+1) for _ in range(X+1)]
    dp[0][0] = 0
    for idx in range(N):
        A_i, B_i = dishes[idx]
        for s in range(X, -1, -1):
            for t in range(Y, -1, -1):
                if dp[s][t] >= 0:
                    new_s = s + A_i
                    new_t = t + B_i
                    if new_s <= X and new_t <= Y:
                        if dp[new_s][new_t] < dp[s][t] +1:
                            dp[new_s][new_t] = dp[s][t] +1
    max_k = 0
    for s in range(X+1):
        for t in range(Y+1):
            if dp[s][t] > max_k:
                max_k = dp[s][t]
    print(max_k)
threading.Thread(target=main).start()