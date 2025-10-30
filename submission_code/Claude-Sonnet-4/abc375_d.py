# YOUR CODE HERE
S = input().strip()
n = len(S)

count = 0

# For each possible middle position j (1-indexed becomes 0-indexed)
for j in range(1, n-1):  # j from 1 to n-2 in 0-indexed
    # Count frequency of each character before position j
    before = {}
    for i in range(j):
        char = S[i]
        before[char] = before.get(char, 0) + 1
    
    # Count frequency of each character after position j  
    after = {}
    for k in range(j+1, n):
        char = S[k]
        after[char] = after.get(char, 0) + 1
    
    # For each character, multiply count before × count after
    for char in before:
        if char in after:
            count += before[char] * after[char]

print(count)