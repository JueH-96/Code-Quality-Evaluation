import sys
import threading
def main():
    import sys, math
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    try:
        n = int(next(it))
    except StopIteration:
        return
    M = int(next(it))
    P = [int(next(it)) for _ in range(n)]
    # If we cannot afford even one of the cheapest product, answer is 0
    pmin = min(P)
    if M < pmin:
        print(0)
        return
    # Compute an upper bound on threshold T:
    # For each product i, the max k_i we could buy alone is k = floor(sqrt(M/P_i))
    # The marginal cost of the k-th unit is (2*k-1)*P_i. The optimum threshold
    # of the S-th unit in the overall solution is at most the maximum of these.
    high = 0
    Mi = M
    # Compute high = max_i (2*isqrt(M//P[i]) -1)*P[i]
    for pi in P:
        # Only if M >= pi does k >= 1
        if Mi >= pi:
            km = math.isqrt(Mi // pi)
            if km > 0:
                Ti = (2*km - 1) * pi
                if Ti > high:
                    high = Ti
    # If no product gives any unit, answer 0
    if high <= 0:
        print(0)
        return
    # Binary search on threshold T in [0, high]
    lo = 0
    hi = high
    # We will test mid: can we pick all units with marginal cost <= mid
    # such that total cost <= M?
    P_list = P  # local alias
    Mval = M
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cost = 0
        # accumulate cost of all units with marginal cost <= mid
        for pi in P_list:
            # For product i, the highest unit index k that has marginal cost <= mid
            # solves pi*(2*k -1) <= mid  => 2*k -1 <= mid/pi  => k <= (mid/pi +1)/2
            v = mid // pi
            if v <= 0:
                continue
            ti = (v + 1) // 2
            # add cost of ti units: pi * ti^2
            cost += pi * ti * ti
            if cost > Mval:
                break
        if cost <= Mval:
            # we can afford all units with marginal cost <= mid
            lo = mid
        else:
            hi = mid - 1
    # lo is the largest threshold we can afford; compute total units
    threshold = lo
    total_units = 0
    for pi in P_list:
        v = threshold // pi
        if v <= 0:
            continue
        total_units += (v + 1) // 2
    print(total_units)

if __name__ == "__main__":
    main()