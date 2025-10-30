def main():
    # Read input
    N, M = map(int, input().split())
    P = list(map(int, input().split()))
    L = list(map(int, input().split()))
    D = list(map(int, input().split()))
    
    print(min_total_cost(N, M, P, L, D))

def min_total_cost(N, M, P, L, D):
    # Calculate potential discount for each coupon-item pair
    discounts = []
    for i in range(N):  # Item index
        for j in range(M):  # Coupon index
            if P[i] >= L[j]:  # Coupon can be applied
                discounts.append((D[j], i, j))  # (discount, item_index, coupon_index)
    
    # Sort by discount in descending order
    discounts.sort(reverse=True)
    
    used_items = set()
    used_coupons = set()
    total_cost = sum(P)  # Start with the total regular cost
    
    for discount, item_idx, coupon_idx in discounts:
        if item_idx not in used_items and coupon_idx not in used_coupons:
            total_cost -= discount  # Apply the discount
            used_items.add(item_idx)
            used_coupons.add(coupon_idx)
    
    return total_cost

if __name__ == "__main__":
    main()