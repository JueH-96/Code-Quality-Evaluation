def main():
    import sys
    S = sys.stdin.read().strip()
    n = len(S)
    
    max_length = 1  # Every single character is a palindrome
    
    for center in range(n):
        # Check for odd-length palindromes (center is a single character)
        left = center
        right = center
        while left >= 0 and right < n and S[left] == S[right]:
            current_length = right - left + 1
            if current_length > max_length:
                max_length = current_length
            left -= 1
            right += 1

        # Check for even-length palindromes (center is between two characters)
        left = center
        right = center + 1
        while left >= 0 and right < n and S[left] == S[right]:
            current_length = right - left + 1
            if current_length > max_length:
                max_length = current_length
            left -= 1
            right += 1

    print(max_length)

if __name__ == "__main__":
    main()