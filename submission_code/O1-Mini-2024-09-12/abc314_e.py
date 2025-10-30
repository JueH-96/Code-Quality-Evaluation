# YOUR CODE HERE
import sys
import math
from decimal import Decimal, getcontext

def main():
    import sys
    import sys
    sys.setrecursionlimit(1000000)
    from decimal import Decimal, getcontext
    getcontext().prec = 50

    N, M = map(int, sys.stdin.readline().split())
    wheels = []
    for _ in range(N):
        parts = sys.stdin.readline().split()
        C_i = Decimal(parts[0])
        P_i = int(parts[1])
        S_i = list(map(int, parts[2:2+P_i]))
        wheels.append( (C_i, P_i, S_i) )
    
    f = [Decimal(0) for _ in range(M+1)]
    # f[M] = 0

    for x in range(M-1, -1, -1):
        min_cost = None
        for wheel in wheels:
            C_i, P_i, S_i = wheel
            total = Decimal(0)
            for s in S_i:
                new_x = x + s
                if new_x >= M:
                    total += Decimal(0)
                else:
                    total += f[new_x]
            average_f = total / Decimal(P_i)
            cost_i = C_i + average_f
            if (min_cost is None) or (cost_i < min_cost):
                min_cost = cost_i
        f[x] = min_cost

    # Print with sufficient decimal places
    print(f[0])

if __name__ == "__main__":
    main()