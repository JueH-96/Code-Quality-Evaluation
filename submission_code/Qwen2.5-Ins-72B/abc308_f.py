# YOUR CODE HERE
import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    P = list(map(int, data[2:N+2]))
    L = list(map(int, data[N+2:N+2+M]))
    D = list(map(int, data[N+2+M:]))
    
    # Create a list of (L_i, D_i) pairs and sort by L_i
    coupons = sorted(zip(L, D), reverse=True)
    
    # Sort the prices in descending order
    prices = sorted(P, reverse=True)
    
    # Initialize the total cost
    total_cost = 0
    
    # Use a pointer to track the current coupon
    coupon_ptr = 0
    
    # Try to apply the coupons to the most expensive items
    for price in prices:
        if coupon_ptr < M and price >= coupons[coupon_ptr][0]:
            total_cost += max(price - coupons[coupon_ptr][1], 0)
            coupon_ptr += 1
        else:
            total_cost += price
    
    print(total_cost)

if __name__ == "__main__":
    solve()