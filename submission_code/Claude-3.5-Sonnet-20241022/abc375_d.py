# Read input
S = input().strip()
n = len(S)

# Function to check if a string is palindrome
def is_palindrome(s):
    return s == s[::-1]

# Count valid triples
count = 0
for i in range(n-2):
    for j in range(i+1, n-1):
        for k in range(j+1, n):
            # Form string by concatenating characters at positions i, j, k
            test_str = S[i] + S[j] + S[k]
            # Check if it forms a palindrome
            if is_palindrome(test_str):
                count += 1

# Print result
print(count)