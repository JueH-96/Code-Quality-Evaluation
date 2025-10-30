# Read the number of strings
N = int(input())

# Initialize the count of Takahashi
count_takahashi = 0

# Read each string and count Takahashi occurrences
for _ in range(N):
    if input().strip() == "Takahashi":
        count_takahashi += 1

# Print the result
print(count_takahashi)