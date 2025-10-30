def can_buy_k_items(k, prices, M):
    """
    Check if it's possible to buy k items in total with cost <= M
    Uses greedy approach: buy from cheapest products first
    """
    if k == 0:
        return True
        
    # Sort prices to buy from cheapest first
    prices = sorted(prices)
    total_cost = 0
    
    # For each price, try to buy as many items as possible
    items_left = k
    for p in prices:
        # If we can't buy anything from this price point, skip
        if items_left == 0:
            break
            
        # Binary search to find how many items we can buy at this price
        left, right = 0, items_left
        best = 0
        
        while left <= right:
            mid = (left + right) // 2
            cost = mid * mid * p
            
            if cost <= M - total_cost:
                best = mid
                left = mid + 1
            else:
                right = mid - 1
        
        total_cost += best * best * p
        items_left -= best
        
        if total_cost > M:
            return False
    
    return items_left == 0

def solve():
    # Read input
    N, M = map(int, input().split())
    prices = list(map(int, input().split()))
    
    # Binary search for the answer
    left, right = 0, 10**18  # Right bound can be reduced but this is safe
    answer = 0
    
    while left <= right:
        mid = (left + right) // 2
        if can_buy_k_items(mid, prices, M):
            answer = mid
            left = mid + 1
        else:
            right = mid - 1
    
    print(answer)

# Run the solution
solve()