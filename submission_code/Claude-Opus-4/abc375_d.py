# YOUR CODE HERE
S = input().strip()
n = len(S)

total = 0

# For each middle position j
for j in range(1, n - 1):  # j can be from index 1 to n-2 (0-indexed)
    # Count occurrences of each character before and after j
    left_count = {}
    right_count = {}
    
    # Count characters before j
    for i in range(j):
        char = S[i]
        left_count[char] = left_count.get(char, 0) + 1
    
    # Count characters after j
    for k in range(j + 1, n):
        char = S[k]
        right_count[char] = right_count.get(char, 0) + 1
    
    # For each character that appears both before and after j
    for char in left_count:
        if char in right_count:
            total += left_count[char] * right_count[char]

print(total)