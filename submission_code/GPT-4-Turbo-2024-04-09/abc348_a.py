# Reading the input
N = int(input())

# Initialize the result string
result = ""

# Loop through each penalty kick
for i in range(1, N + 1):
    if i % 3 == 0:
        result += "x"  # Takahashi fails on multiples of 3
    else:
        result += "o"  # Takahashi succeeds otherwise

# Print the result
print(result)