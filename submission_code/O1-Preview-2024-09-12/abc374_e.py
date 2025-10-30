# YOUR CODE HERE
import sys
import math
import threading
def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, X = map(int, sys.stdin.readline().split())
    A = []
    B = []
    P = []
    Q = []
    for _ in range(N):
        a, p, b, q = map(int, sys.stdin.readline().split())
        A.append(a)
        P.append(p)
        B.append(b)
        Q.append(q)

    Max_S_limit = 1000
    Max_T_limit = 1000

    # Binary search on production capacity W
    low = 0
    high = 10**9  # As per sample input 2

    while low < high:
        mid = (low + high + 1) // 2

        total_cost = 0
        possible = True
        for i in range(N):
            min_cost = float('inf')

            # Check s_i values
            s_min = max(0, (mid - B[i] * Max_T_limit) // A[i])
            s_max = (mid + A[i] - 1) // A[i] + Max_S_limit
            for s in range(s_min, s_max + 1):
                t_numerator = mid - A[i] * s
                if t_numerator <= 0:
                    t = 0
                else:
                    t = (t_numerator + B[i] - 1) // B[i]
                if t < 0:
                    continue
                cost = P[i] * s + Q[i] * t
                if cost < min_cost:
                    min_cost = cost

            # Check t_i values
            t_min = max(0, (mid - A[i] * Max_S_limit) // B[i])
            t_max = (mid + B[i] - 1) // B[i] + Max_T_limit
            for t in range(t_min, t_max + 1):
                s_numerator = mid - B[i] * t
                if s_numerator <= 0:
                    s = 0
                else:
                    s = (s_numerator + A[i] - 1) // A[i]
                if s < 0:
                    continue
                cost = P[i] * s + Q[i] * t
                if cost < min_cost:
                    min_cost = cost
            if min_cost == float('inf'):
                possible = False
                break
            total_cost += min_cost
            if total_cost > X:
                possible = False
                break
        if possible and total_cost <= X:
            low = mid
        else:
            high = mid - 1
    print(low)

if __name__ == '__main__':
    threading.Thread(target=main,).start()