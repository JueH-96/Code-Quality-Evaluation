import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

prices = list(map(int, data[2:N+2]))
coupons = []

index = N + 2
for i in range(M):
    L = int(data[index])
    D = int(data[index + 1])
    coupons.append((L, D))
    index += 2

# Sort prices in descending order
prices.sort(reverse=True)

# Sort coupons by L in descending order, and by D in descending order if L is the same
coupons.sort(reverse=True, key=lambda x: (x[0], x[1]))

# Min-heap to store the best discounts we can apply
discounts = []
total_cost = 0
coupon_index = 0

for price in prices:
    # Push all coupons that can be applied to items priced at least `price` into the heap
    while coupon_index < M and coupons[coupon_index][0] >= price:
        heapq.heappush(discounts, -coupons[coupon_index][1])  # Use negative to simulate max heap
        coupon_index += 1
    
    # If there are applicable discounts, use the best one
    if discounts:
        best_discount = -heapq.heappop(discounts)
        total_cost += max(0, price - best_discount)
    else:
        total_cost += price

print(total_cost)