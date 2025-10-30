def find_max_palindrome_length(S):
    n = len(S)
    max_length = 1  # At least one character is a palindrome

    for i in range(n):
        # Odd length palindromes
        left, right = i, i
        while left >= 0 and right < n and S[left] == S[right]:
            left -= 1
            right += 1
        # Update the max_length if a longer palindrome is found
        current_length = right - left - 1
        if current_length > max_length:
            max_length = current_length

        # Even length palindromes
        left, right = i, i + 1
        while left >= 0 and right < n and S[left] == S[right]:
            left -= 1
            right += 1
        # Update the max_length if a longer palindrome is found
        current_length = right - left - 1
        if current_length > max_length:
            max_length = current_length

    return max_length

# Read input
S = input().strip()
# Find and print the maximum palindrome length
print(find_max_palindrome_length(S))