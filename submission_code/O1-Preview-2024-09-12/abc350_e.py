# YOUR CODE HERE
import sys
import threading
import math
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, A, X, Y = map(int, sys.stdin.readline().split())
    from functools import lru_cache

    N_max = int(1e6)
    ln_A = math.log(A)
    ln_b_list = [math.log(b) for b in range(1, 7)]
    E_ln_b = sum(ln_b_list) / 6

    @lru_cache(maxsize=None)
    def f(n):
        if n == 0:
            return 0.0
        if n <= N_max:
            cost1 = X + f(n // A)
            cost2 = Y + (sum(f(n // b) for b in range(1, 7)) ) / 6.0
            return min(cost1, cost2)
        else:
            # For large n, approximate the cost
            cost1 = X * math.ceil(math.log(n) / ln_A)
            # Expected number of steps for option 2
            # We can approximate the expected total cost for option 2
            expected_cost2 = Y + (sum(f(n // b) for b in range(1, 7)) ) / 6.0
            return min(cost1, expected_cost2)

    ans = f(N)
    print("%.15f" % ans)

threading.Thread(target=main).start()