import sys

# Read input
N, K, X = map(int, input().split())
T = list(map(int, input().split()))

# Sort the order placement days
T.sort()

# Initialize variables
total_dissatisfaction = 0
last_ship_day = 0

# Iterate through the orders
for i in range(N):
    # Find the earliest day to ship the current order
    ship_day = max(T[i], last_ship_day)
    
    # Calculate the dissatisfaction for the current order
    dissatisfaction = ship_day - T[i]
    total_dissatisfaction += dissatisfaction
    
    # Update the last ship day
    last_ship_day = ship_day + X
    
    # Check if we can ship multiple orders together
    if i + 1 < N and T[i+1] <= last_ship_day:
        j = i + 1
        while j < N and j - i < K and T[j] <= last_ship_day:
            j += 1
        i = j - 1
        last_ship_day = ship_day + X

print(total_dissatisfaction)