def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    S = input_data[0]
    n = len(S)
    
    max_len = 1  # any single letter is a palindrome

    # Check for odd-length palindromes (centered at each index)
    for i in range(n):
        left, right = i, i
        while left >= 0 and right < n and S[left] == S[right]:
            max_len = max(max_len, right - left + 1)
            left -= 1
            right += 1

    # Check for even-length palindromes (center between indices)
    for i in range(n - 1):
        left, right = i, i + 1
        while left >= 0 and right < n and S[left] == S[right]:
            max_len = max(max_len, right - left + 1)
            left -= 1
            right += 1

    sys.stdout.write(str(max_len))

if __name__ == '__main__':
    main()