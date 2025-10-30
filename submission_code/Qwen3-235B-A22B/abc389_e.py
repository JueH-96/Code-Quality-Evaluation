import sys

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    P.sort()

    def compute_sum_count(T):
        res = 0
        for p in P:
            if T < p:
                break
            q = T // p
            k = (q + 1) // 2
            res += k
        return res

    def compute_sum_total(T):
        res = 0
        for p in P:
            if T < p:
                break
            q = T // p
            k = (q + 1) // 2
            res += k * k * p
        return res

    def compute_min_cost(X):
        if X == 0:
            return 0
        low = 0
        high = 1
        # Expand high until compute_sum_count(high) >= X
        while compute_sum_count(high) < X:
            high *= 2
        # Binary search for T
        while low < high:
            mid = (low + high) // 2
            cnt = compute_sum_count(mid)
            if cnt < X:
                low = mid + 1
            else:
                high = mid
        T = low
        # Calculate sum_prev_count and sum_prev_total
        sum_prev_count = compute_sum_count(T - 1)
        sum_prev_total = compute_sum_total(T - 1)
        needed = X - sum_prev_count
        total = sum_prev_total + needed * T
        return total

    # Edge case when X=0
    if compute_min_cost(1) > M:
        print(0)
        return

    # Find upper bound for X
    low_x = 0
    high_x = 1
    while True:
        cost = compute_min_cost(high_x)
        if cost <= M:
            high_x *= 2
            if high_x > 1e18:
                break
        else:
            break

    # Binary search for the maximum X
    while low_x < high_x:
        mid = (low_x + high_x + 1) // 2
        cost = compute_min_cost(mid)
        if cost <= M:
            low_x = mid
        else:
            high_x = mid - 1

    print(low_x)

if __name__ == '__main__':
    main()