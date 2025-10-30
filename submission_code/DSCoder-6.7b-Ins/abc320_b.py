def longest_palindrome(s):
    def get_palindrome(s, i, j):
        while i >= 0 and j < len(s) and s[i] == s[j]:
            i -= 1
            j += 1
        return s[i+1:j]

    longest = ''
    for k in range(len(s)):
        # odd length palindrome
        palindrome_odd = get_palindrome(s, k, k)
        if len(palindrome_odd) > len(longest):
            longest = palindrome_odd
        # even length palindrome
        palindrome_even = get_palindrome(s, k, k+1)
        if len(palindrome_even) > len(longest):
            longest = palindrome_even

    return len(longest)

S = input()
print(longest_palindrome(S))