# Read the number of penalty kicks from stdin
N = int(input())

# Initialize an empty string to store the results
results = ""

# Loop through each penalty kick
for i in range(1, N + 1):
    # If the kick number is a multiple of 3, Takahashi fails
    if i % 3 == 0:
        results += "x"
    # Otherwise, Takahashi succeeds
    else:
        results += "o"

# Print the results
print(results)