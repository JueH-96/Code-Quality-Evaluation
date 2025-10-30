# YOUR CODE HERE
def minimum_payment(N, P, Q, D):
    # Calculate the minimum cost if using the coupon
    min_dish_price = min(D)
    total_with_coupon = Q + min_dish_price
    
    # The minimum total amount he must pay
    return min(total_with_coupon, P)

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = int(data[1])
Q = int(data[2])
D = list(map(int, data[3:]))

# Get the result and print it
result = minimum_payment(N, P, Q, D)
print(result)