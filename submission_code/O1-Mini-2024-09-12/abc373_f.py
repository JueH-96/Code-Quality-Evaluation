import numpy as np
import sys

def solve():
    import sys
    import numpy as np
    N, W = map(int, sys.stdin.readline().split())
    types = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    dp = np.full(W +1, -1e18)
    dp[0] = 0
    for w_i, v_i in types:
        if w_i > W:
            continue
        k_max = min(W //w_i, (v_i +1) //2 )
        if k_max ==0:
            continue
        k = np.arange(1, k_max +1)
        weights = k *w_i
        happiness = k *v_i -k *k
        temp = dp.copy()
        for weight, h in zip(weights, happiness):
            if weight > W:
                continue
            temp[weight:] = np.maximum(temp[weight:], dp[:W +1 -weight] + h)
        dp = temp
    print(int(np.max(dp)))