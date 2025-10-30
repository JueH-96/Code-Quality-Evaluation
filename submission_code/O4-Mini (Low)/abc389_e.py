import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    P = [int(next(it)) for _ in range(N)]
    
    # Given a threshold T on marginal cost, for each product i
    # we can buy k_i = floor((T/P_i + 1) / 2) units.
    # Let C(T) = sum_i P_i * k_i^2 be the total cost under that rule.
    # C(T) is non-decreasing in T, and the total units S(T)=sum k_i is non-decreasing.
    # We binary‐search the largest T such that C(T) <= M. Then S(T) is the maximum units.
    
    def compute_cost_and_units(T):
        # returns (C, S)
        total_cost = 0
        total_units = 0
        # we can bail out early if cost exceeds M
        for p in P:
            # k = floor((T/p + 1)/2)
            k = (T // p + 1) // 2
            if k:
                total_units += k
                total_cost += p * (k * k)
                if total_cost > M:
                    # no need to continue
                    return total_cost, total_units
        return total_cost, total_units
    
    lo = 0
    hi = 2 * M  # upper bound on marginal cost
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cost_mid, _ = compute_cost_and_units(mid)
        if cost_mid <= M:
            lo = mid
        else:
            hi = mid - 1
    
    # lo is the maximum marginal threshold we can afford
    _, answer = compute_cost_and_units(lo)
    print(answer)

if __name__ == "__main__":
    main()