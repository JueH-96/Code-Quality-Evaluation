import sys

# Read input
N, X, Y = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Initialize variables
total_sweetness = 0
total_saltiness = 0
dishes_eaten = 0

# Iterate through the dishes
for i in range(N):
    # Add the sweetness and saltiness of the current dish
    total_sweetness += A[i]
    total_saltiness += B[i]
    dishes_eaten += 1
    
    # Check if the total sweetness or saltiness exceeds the limits
    if total_sweetness > X or total_saltiness > Y:
        break

# Print the minimum number of dishes eaten
print(dishes_eaten)