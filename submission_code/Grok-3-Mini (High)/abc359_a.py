# Read N from input
N = int(input())

# Initialize counter for Takahashi occurrences
count = 0

# Read N strings and count how many are "Takahashi"
for _ in range(N):
    s = input().strip()
    if s == "Takahashi":
        count += 1

# Output the count
print(count)