# YOUR CODE HERE
S = input().strip()
n = len(S)
max_length = 1  # At minimum, any single character is a palindrome

# Check all possible substrings
for i in range(n):
    for j in range(i + 1, n + 1):
        substring = S[i:j]
        # Check if substring is a palindrome
        if substring == substring[::-1]:
            max_length = max(max_length, len(substring))

print(max_length)