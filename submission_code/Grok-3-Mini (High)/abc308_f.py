import sys
from sortedcontainers import SortedList

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2
P_list = list(map(int, data[index:index + N]))
index += N
L_list = list(map(int, data[index:index + M]))
index += M
D_list = list(map(int, data[index:index + M]))

# Compute sum of all prices
sum_P = sum(P_list)

# Create a sorted list of available item prices
sl = SortedList(P_list)  # Sorted in increasing order

# Create a list of coupons and sort by discount decreasing
coupons = sorted(zip(D_list, L_list), reverse=True)  # Each coupon is (D, L)

# Initialize discount
discount = 0

# For each coupon in decreasing order of discount
for D_c, L_c in coupons:
    # Find the index where price >= L_c using bisect_left
    idx = sl.bisect_left(L_c)
    # If there is an item with price >= L_c
    if idx < len(sl) and sl[idx] >= L_c:
        # Add the discount
        discount += D_c
        # Remove the assigned item from the sorted list
        sl.pop(idx)

# Calculate minimum cost
min_cost = sum_P - discount

# Output the result
print(min_cost)