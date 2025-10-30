def is_palindrome(s):
    return s == s[::-1]

def check_palindrome_pairs(n, strings):
    for i in range(n):
        for j in range(n):
            if i != j and is_palindrome(strings[i] + strings[j]):
                return "Yes"
    return "No"

# Read input from stdin
n = int(input().strip())
strings = [input().strip() for _ in range(n)]

# Check for palindrome pairs and print the result to stdout
print(check_palindrome_pairs(n, strings))