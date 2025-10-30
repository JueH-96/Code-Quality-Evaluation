# YOUR CODE HERE
N = int(input())

# Calculate the nearest water station
nearest = round(N / 5) * 5

# Ensure the result is within the valid range (0 to 100)
nearest = max(0, min(100, nearest))

print(nearest)