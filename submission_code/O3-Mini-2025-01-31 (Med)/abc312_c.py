def main():
    import sys
    import bisect
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    sellers = [int(next(it)) for _ in range(N)]
    buyers = [int(next(it)) for _ in range(M)]
    
    # Sort seller minimum prices and buyer maximum prices.
    sellers.sort()  # sellers: lowest first
    buyers.sort()   # buyers: lowest first

    # The problem asks:
    # Find the smallest integer X such that the number of sellers who can sell at X yen (seller with A_i <= X)
    # is greater than or equal to the number of buyers who can buy at X yen (buyer with B_i >= X).
    #
    # To compute these quickly:
    # For sellers: number of sellers with A_i <= X is given by bisect_right(sellers, X).
    # For buyers: number of buyers with B_i >= X is M - bisect_left(buyers, X).

    # Let f(X) = (count_sellers - count_buyers). As X increases, f(X) is non-decreasing.
    # We want minimal X such that f(X) >= 0.

    # The search domain for X can be set as follows:
    # Lower bound: at least 1.
    # Upper bound: buyers[-1] + 1, because if X > max(B), no buyer will be able to buy (buyer count = 0),
    # and the condition (seller count >= 0) will be trivially true.
    
    lo = 1
    hi = buyers[-1] + 1  # Since buyers are sorted in ascending order.

    while lo < hi:
        mid = (lo + hi) // 2
        seller_count = bisect.bisect_right(sellers, mid)  # Sellers who accept mid yen.
        buyer_count = M - bisect.bisect_left(buyers, mid)   # Buyers who can pay at least mid yen.
        if seller_count >= buyer_count:
            hi = mid
        else:
            lo = mid + 1

    print(lo)

if __name__ == '__main__':
    main()