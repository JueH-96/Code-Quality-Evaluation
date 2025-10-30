def find_minimum_price(A, B):
    left = 1
    right = 10**9
    
    while left <= right:
        mid = (left + right) // 2
        
        # Count sellers willing to sell at price mid
        willing_sellers = sum(1 for a in A if mid >= a)
        
        # Count buyers willing to buy at price mid
        willing_buyers = sum(1 for b in B if mid <= b)
        
        if willing_sellers >= willing_buyers:
            # This price satisfies the condition, but we want the minimum
            # so try to find a lower price
            right = mid - 1
        else:
            # This price doesn't satisfy the condition
            # need to look for a higher price
            left = mid + 1
    
    return left

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Find and print the answer
print(find_minimum_price(A, B))