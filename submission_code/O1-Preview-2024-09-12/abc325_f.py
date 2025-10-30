# YOUR CODE HERE
import sys
import math

import threading
def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N = int(sys.stdin.readline())
    D = list(map(int, sys.stdin.readline().split()))
    L1, C1, K1 = map(int, sys.stdin.readline().split())
    L2, C2, K2 = map(int, sys.stdin.readline().split())

    N = len(D)

    # Generate per D_i feasible options
    options = []
    for D_i in D:
        opt_i = []
        max_n1 = min(K1, (D_i + L1 - 1) // L1)
        for n1 in range(max_n1 + 1):
            # Remaining length to cover
            rem = D_i - n1 * L1
            if rem <= 0:
                cost = n1 * C1
                n2 = 0
                opt_i.append((n1, n2, cost))
                continue
            # Need to cover rem with type 2 sensors
            min_n2 = (rem + L2 - 1) // L2
            if min_n2 > K2:
                continue
            cost = n1 * C1 + min_n2 * C2
            n2 = min_n2
            if n2 > K2:
                continue
            opt_i.append((n1, n2, cost))
        # Similarly, try n2 from 0 to max_n2
        max_n2 = min(K2, (D_i + L2 -1) // L2)
        for n2 in range(max_n2 +1):
            rem = D_i - n2 * L2
            if rem <=0:
                cost = n2 * C2
                n1 = 0
                opt_i.append((n1, n2, cost))
                continue
            min_n1 = (rem + L1 -1) // L1
            if min_n1 > K1:
                continue
            cost = n2 * C2 + min_n1 * C1
            n1 = min_n1
            if n1 > K1:
                continue
            opt_i.append((n1, n2, cost))
        # Remove duplicate options
        opt_set = set()
        opt_unique = []
        for o in opt_i:
            if (o[0], o[1]) not in opt_set:
                opt_set.add((o[0], o[1]))
                opt_unique.append(o)
        options.append(opt_unique)

    dp = {}
    dp[(0, 0)] = 0  # (total_n1, total_n2): total_cost

    for idx in range(N):
        opt_i = options[idx]
        dp_new = {}
        for (k1, k2), cost in dp.items():
            for n1_i, n2_i, c_i in opt_i:
                k1_new = k1 + n1_i
                k2_new = k2 + n2_i
                if k1_new > K1 or k2_new > K2:
                    continue
                total_cost = cost + c_i
                key = (k1_new, k2_new)
                if key not in dp_new or dp_new[key] > total_cost:
                    dp_new[key] = total_cost
        dp = dp_new
        if not dp:
            print(-1)
            return

    if not dp:
        print(-1)
    else:
        min_total_cost = min(dp.values())
        print(min_total_cost)




threading.Thread(target=main,).start()