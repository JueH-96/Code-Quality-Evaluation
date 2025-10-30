# Read the number of strings
N = int(input())

# Initialize count of "Takahashi"
count = 0

# Loop through each string
for _ in range(N):
    S = input().strip()
    if S == "Takahashi":
        count += 1

# Print the result
print(count)