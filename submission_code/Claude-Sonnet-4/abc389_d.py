# YOUR CODE HERE
R = int(input())

count = 0
R_squared = R * R

# For each possible value of i
for i in range(int(R) + 1):
    # Find maximum j such that (i+0.5)^2 + (j+0.5)^2 <= R^2
    i_term = (i + 0.5) ** 2
    if i_term > R_squared:
        break
    
    remaining = R_squared - i_term
    max_j = int((remaining ** 0.5) - 0.5)
    
    if max_j >= 0:
        # Count squares for this i value
        if i == 0:
            # On the y-axis: count from -max_j to max_j
            count += 2 * max_j + 1
        else:
            # Not on y-axis: count positive and negative i values
            # For each i value (positive and negative), count from -max_j to max_j
            count += 2 * (2 * max_j + 1)

print(count)