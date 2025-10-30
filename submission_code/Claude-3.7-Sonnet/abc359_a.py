# Read the number of strings
n = int(input())

# Count the number of times "Takahashi" appears
takahashi_count = 0
for _ in range(n):
    s = input()
    if s == "Takahashi":
        takahashi_count += 1

# Print the count
print(takahashi_count)