# YOUR CODE HERE

# Read the inputs
S_x, S_y = map(int, input().split())
T_x, T_y = map(int, input().split())

# Calculate the differences
diff_x = abs(S_x - T_x)
diff_y = abs(S_y - T_y)

# Calculate the minimum toll
min_toll = min(diff_x, diff_y)

# Print the minimum toll
print(min_toll)