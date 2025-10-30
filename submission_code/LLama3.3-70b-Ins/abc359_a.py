# Read the number of strings
N = int(input())

# Initialize a counter for "Takahashi"
takahashi_count = 0

# Read each string and count "Takahashi"
for _ in range(N):
    S = input()
    if S == "Takahashi":
        takahashi_count += 1

# Print the count of "Takahashi"
print(takahashi_count)