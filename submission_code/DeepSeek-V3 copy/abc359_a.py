# Read the number of strings
N = int(input())
count = 0
# Iterate through each string
for _ in range(N):
    S = input().strip()
    if S == "Takahashi":
        count += 1
# Print the count
print(count)