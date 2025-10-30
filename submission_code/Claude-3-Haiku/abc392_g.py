from collections import defaultdict

# Read the input
N = int(input())
S = [int(x) for x in input().split()]

# Create a dictionary to store the differences between elements
diff_count = defaultdict(int)

# Count the number of fine triplets
count = 0
for i in range(N):
    for j in range(i+1, N):
        diff = S[j] - S[i]
        if diff in diff_count:
            count += diff_count[diff]
        diff_count[diff] += 1

print(count)