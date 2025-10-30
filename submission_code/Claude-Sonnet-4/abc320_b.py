# YOUR CODE HERE
S = input().strip()
n = len(S)

def is_palindrome(s):
    return s == s[::-1]

max_length = 1  # At least one character is always a palindrome

# Check all possible substring lengths from longest to shortest
for length in range(n, 0, -1):
    # Check all substrings of this length
    for start in range(n - length + 1):
        substring = S[start:start + length]
        if is_palindrome(substring):
            max_length = length
            break
    if max_length == length:  # Found a palindrome of this length
        break

print(max_length)