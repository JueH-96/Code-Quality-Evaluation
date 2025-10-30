# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    X = []
    Y = []
    for _ in range(N):
        x_i, y_i = map(int, sys.stdin.readline().split())
        X.append(x_i)
        Y.append(y_i)
    NEG_INF = -1 << 60
    DP_prev = [NEG_INF, NEG_INF]  # DP_prev[0]: healthy, DP_prev[1]: upset
    DP_prev[0] = 0  # Start with healthy stomach and total tastiness 0
    for i in range(N):
        x_i, y_i = X[i], Y[i]
        DP_curr = [NEG_INF, NEG_INF]
        # Option 1: skip the course
        for s in [0, 1]:
            DP_curr[s] = max(DP_curr[s], DP_prev[s])
        # Option 2: eat the course
        # From healthy state
        if DP_prev[0] > NEG_INF:
            if x_i == 0:
                # Eating antidotal course, remains healthy
                DP_curr[0] = max(DP_curr[0], DP_prev[0] + y_i)
            elif x_i == 1:
                # Eating poisonous course, becomes upset
                DP_curr[1] = max(DP_curr[1], DP_prev[0] + y_i)
        # From upset state
        if DP_prev[1] > NEG_INF:
            if x_i == 0:
                # Eating antidotal course, becomes healthy
                DP_curr[0] = max(DP_curr[0], DP_prev[1] + y_i)
            elif x_i == 1:
                # Eating poisonous course, dies - cannot proceed
                pass
        DP_prev = DP_curr
    result = max(DP_prev[0], DP_prev[1])
    if result > NEG_INF:
        print(result)
    else:
        print(0)

threading.Thread(target=main,).start()