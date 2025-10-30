# YOUR CODE HERE

def longest_palindrome(s):
    def is_palindrome(s):
        return s == s[::-1]

    max_length = 0
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            if is_palindrome(s[i:j]) and j - i > max_length:
                max_length = j - i
    return max_length

s = input()
print(longest_palindrome(s))