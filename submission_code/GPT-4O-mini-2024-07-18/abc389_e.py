def max_units(N, M, P):
    # Sort the prices to prioritize cheaper products
    P.sort()
    
    total_units = 0
    remaining_budget = M
    
    for price in P:
        # Calculate the maximum units we can buy for this product
        # We need to solve k^2 * price <= remaining_budget
        # which gives us k^2 <= remaining_budget / price
        # thus k <= sqrt(remaining_budget / price)
        
        max_k = int((remaining_budget // price) ** 0.5)
        
        # Calculate the cost for max_k units
        cost = max_k * max_k * price
        
        # If we can afford this cost, we buy max_k units
        if cost <= remaining_budget:
            total_units += max_k
            remaining_budget -= cost
        else:
            # If we can't afford max_k, we need to find the maximum k we can afford
            # We can use a binary search to find the maximum k that fits in the budget
            low, high = 0, max_k
            
            while low < high:
                mid = (low + high + 1) // 2
                if mid * mid * price <= remaining_budget:
                    low = mid  # mid is affordable
                else:
                    high = mid - 1  # mid is too expensive
            
            total_units += low  # low is the maximum k we can afford
            break  # No more budget left to buy any more products
    
    return total_units

import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    M = int(data[1])
    P = list(map(int, data[2:2 + N]))
    
    result = max_units(N, M, P)
    print(result)

if __name__ == "__main__":
    main()