# Read input
N = int(input())
A = list(map(int, input().split()))

# Count frequency of each number
freq = {}
for num in A:
    freq[num] = freq.get(num, 0) + 1

# Find the person with largest unique number
max_unique = -1
max_unique_person = -1

# Check each person
for i in range(N):
    # If this person's number appears exactly once
    if freq[A[i]] == 1:
        # If this is the largest unique number we've seen
        if A[i] > max_unique:
            max_unique = A[i]
            max_unique_person = i + 1  # Add 1 because people are labeled 1 to N

# Print result
print(max_unique_person)