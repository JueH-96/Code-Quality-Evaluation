import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    P = list(map(int, data[2:]))

    # We want to find the maximum total units T we can buy
    # so that sum of the T smallest marginal costs <= M.
    # Marginal costs for product i are P[i] * (2*j - 1) for j = 1,2,3,...
    # Define for a threshold x:
    #   k_i(x) = number of units of i whose marginal cost <= x
    #          = max j: P[i]*(2*j-1) <= x
    #          = floor((x/P[i] + 1)/2).
    # Then total units K(x) = sum_i k_i(x)
    # and cost S(x) = sum_i P[i] * k_i(x)^2.
    # Both K(x) and S(x) are non-decreasing in x.
    # We binary‐search on x to find the largest x such that S(x) <= M,
    # and then answer = K(x).

    lo = 0
    hi = 2 * M + 1  # any margin larger than max needed
    bestK = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        total_k = 0
        total_cost = 0
        # compute K(mid) and S(mid)
        for p in P:
            if mid < p:
                continue
            # k_i = floor((mid // p + 1) / 2)
            ki = (mid // p + 1) // 2
            total_k += ki
            # add p * ki^2, but cut off early if we exceed M
            total_cost += p * ki * ki
            if total_cost > M:
                break

        if total_cost <= M:
            # we can afford all units with marginal cost <= mid
            bestK = total_k
            lo = mid + 1
        else:
            hi = mid - 1

    # bestK is the maximum total units whose marginal costs fit in M
    print(bestK)

if __name__ == "__main__":
    main()