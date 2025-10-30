def main():
    import sys
    S = sys.stdin.readline().strip()

    n = len(S)
    max_len = 1  # We know there's always at least a length-1 palindrome

    # Check for odd-length palindromes
    for center in range(n):
        left, right = center, center
        while left >= 0 and right < n and S[left] == S[right]:
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
            left -= 1
            right += 1

    # Check for even-length palindromes
    for center in range(n-1):
        left, right = center, center + 1
        while left >= 0 and right < n and S[left] == S[right]:
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
            left -= 1
            right += 1

    print(max_len)

if __name__ == "__main__":
    main()