# YOUR CODE HERE
n = int(input())
s = input().strip()

# Dictionary to store the maximum consecutive length for each character
max_consecutive = {}

i = 0
while i < n:
    char = s[i]
    count = 1
    
    # Count consecutive occurrences of the same character
    while i + count < n and s[i + count] == char:
        count += 1
    
    # Update maximum consecutive length for this character
    if char not in max_consecutive:
        max_consecutive[char] = count
    else:
        max_consecutive[char] = max(max_consecutive[char], count)
    
    i += count

# Calculate total number of distinct substrings
total = 0
for char, max_len in max_consecutive.items():
    total += max_len

print(total)