# YOUR CODE HERE
# Read input
line1 = input().split()
N = int(line1[0])
P = int(line1[1])
Q = int(line1[2])

line2 = input().split()
D = [int(x) for x in line2]

# Calculate minimum cost
regular_cost = P
coupon_cost = Q + min(D)

answer = min(regular_cost, coupon_cost)
print(answer)