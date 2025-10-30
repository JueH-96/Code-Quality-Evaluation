def main():
    import sys
    S = sys.stdin.readline().strip()
    n = len(S)
    # function to expand around center i, j
    def expand(i, j):
        while i >= 0 and j < n and S[i] == S[j]:
            i -= 1
            j += 1
        return j - i - 1  # length of the palindrome
    
    max_length = 0
    for i in range(n):
        # for odd-length palindromes (center at i)
        len1 = expand(i, i)
        # for even-length palindromes (centers at i and i+1)
        len2 = expand(i, i + 1)
        current_max = max(len1, len2)
        if current_max > max_length:
            max_length = current_max
    print(max_length)

if __name__ == '__main__':
    main()