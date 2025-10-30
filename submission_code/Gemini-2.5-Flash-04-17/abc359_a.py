# Read the number of strings
N = int(input())

# Initialize a counter for "Takahashi"
takahashi_count = 0

# Loop N times to read each string
for _ in range(N):
    s = input()
    # Check if the current string is "Takahashi"
    if s == "Takahashi":
        takahashi_count += 1

# Print the final count
print(takahashi_count)