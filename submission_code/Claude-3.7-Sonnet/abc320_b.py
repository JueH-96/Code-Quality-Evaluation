def longest_palindrome(s):
    if not s:
        return 0

    n = len(s)
    max_length = 1  # A single character is always a palindrome

    # Function to expand outwards from the center
    def expand_from_center(left, right):
        while left >= 0 and right < n and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

    for i in range(n):
        # Expand odd length palindromes (single character center)
        odd_length = expand_from_center(i, i)
        max_length = max(max_length, odd_length)

        # Expand even length palindromes (between two characters)
        if i + 1 < n:
            even_length = expand_from_center(i, i + 1)
            max_length = max(max_length, even_length)

    return max_length

s = input().strip()
print(longest_palindrome(s))