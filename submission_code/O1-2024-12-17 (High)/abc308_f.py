def main():
    import sys
    import bisect
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    P = list(map(int, data[2:2+N]))
    L = list(map(int, data[2+N:2+N+M]))
    D = list(map(int, data[2+N+M:2+N+2*M]))
    
    # 1) Compute the total sum of item prices.
    total_price = sum(P)
    
    # 2) Sort all item prices in ascending order.
    P_sorted = sorted(P)
    
    # 3) Pair each coupon as (discount, minimum_price_requirement) 
    #    and sort them by discount in descending order.
    coupons = [(D[i], L[i]) for i in range(M)]
    coupons.sort(key=lambda x: x[0], reverse=True)
    
    # 4) We will greedily match coupons (from largest discount to smallest) 
    #    to the "smallest" item whose price is >= L. Once an item is used, 
    #    it cannot be used again. We maintain a "next" array that helps us 
    #    quickly find the next unused item index.
    nxt = list(range(N+1))  # nxt[i] points to the next unused item index after i
    
    def find(x):
        """Path compression find: returns the next unused item index >= x."""
        if nxt[x] != x:
            nxt[x] = find(nxt[x])
        return nxt[x]
    
    total_discount = 0
    
    # 5) For each coupon in order of descending discount:
    #    - Use binary search to find the leftmost item index i s.t. P_sorted[i] >= coupon's L
    #    - Then jump to find(i) to skip over any used items
    #    - If valid, apply the discount and mark the item as used by linking it to find(i+1)
    import bisect
    for discount, requirement in coupons:
        i = bisect.bisect_left(P_sorted, requirement)
        if i == N:        # No item has price >= requirement
            continue
        i2 = find(i)
        if i2 == N:       # No more unused items from position i onward
            continue
        total_discount += discount
        nxt[i2] = find(i2+1)  # Mark this item as used
    
    # 6) The answer is the total price minus the total discount we managed to apply
    print(total_price - total_discount)

# Do not forget to call main()
if __name__ == "__main__":
    main()