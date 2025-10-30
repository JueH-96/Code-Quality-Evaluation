# Read the input string and strip any whitespace
num_str = input().strip()

# Check if all adjacent digits are strictly decreasing
is_decreasing = all(int(num_str[i]) > int(num_str[i + 1]) for i in range(len(num_str) - 1))

# Print Yes if it is strictly decreasing, otherwise No
print("Yes" if is_decreasing else "No")