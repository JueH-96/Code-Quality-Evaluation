# YOUR CODE HERE
N, M, P = map(int, input().split())

# Calculate the number of full moons he can see from day 1 to day N
full_moons = (N - M) // P + 1

# Ensure the result is non-negative
full_moons = max(0, full_moons)

print(full_moons)