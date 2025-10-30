def main():
    import sys
    import bisect

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    # Read seller minimum prices and buyer maximum prices.
    sellers = [int(next(it)) for _ in range(n)]
    buyers  = [int(next(it)) for _ in range(m)]
    
    # Sort the sellers’ asking prices and buyers’ bid prices.
    sellers.sort()
    buyers.sort()
    
    # For a given price X, the i-th seller is willing to sell if X >= A_i,
    # i.e. the number of sellers willing to sell at X is the count of sellers
    # with A_i <= X.
    # Similarly, the i-th buyer is willing to buy if X <= B_i,
    # i.e. the number of buyers willing to buy at X is the count of 
    # buyers with B_i >= X.
    #
    # We want the minimum integer price X such that:
    #   count_sellers (with A_i <= X) >= count_buyers (with B_i >= X)
    #
    # Notice:
    # - count_sellers is non-decreasing in X.
    # - count_buyers is non-increasing in X.
    #
    # We can use binary search on X.
    
    # Set lower bound and upper bound.
    # Lower bound: 1 (since prices are at least 1)
    # Upper bound: one more than the maximum value among A's and B's.
    lo = 1
    hi = max(max(sellers), max(buyers)) + 1
    ans = hi  # initialize answer
    
    while lo <= hi:
        mid = (lo + hi) // 2
        # Count sellers who can sell for mid yen.
        # bisect_right gives the index of the first element > mid.
        count_sellers = bisect.bisect_right(sellers, mid)
        # Count buyers who can buy for mid yen.
        # bisect_left gives the index of the first element not less than mid.
        count_buyers = m - bisect.bisect_left(buyers, mid)
        
        if count_sellers >= count_buyers:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1

    sys.stdout.write(str(ans))


if __name__ == '__main__':
    main()