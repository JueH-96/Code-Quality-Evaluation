def find_minimum_price(N, M, sellers, buyers):
    sellers.sort()
    buyers.sort()
    
    left, right = 0, 10**9 + 1  # X can range from 0 to 10^9
    
    while left < right:
        mid = (left + right) // 2
        
        # Count sellers who can sell for at least mid
        count_sellers = sum(1 for price in sellers if price <= mid)
        # Count buyers who can buy for at most mid
        count_buyers = sum(1 for price in buyers if price >= mid)
        
        if count_sellers >= count_buyers:
            right = mid  # Try for a smaller price
        else:
            left = mid + 1  # Increase the price
    
    return left

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
sellers = list(map(int, data[2:N+2]))
buyers = list(map(int, data[N+2:N+2+M]))

result = find_minimum_price(N, M, sellers, buyers)
print(result)