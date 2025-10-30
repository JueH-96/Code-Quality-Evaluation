import math

# Read Takahashi's height
H = int(input().strip())

# Calculate the first day the plant is taller than Takahashi
result = math.ceil(math.log2(H + 2))

# Print the result
print(result)