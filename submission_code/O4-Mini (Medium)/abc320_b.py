def main():
    S = input().strip()
    n = len(S)
    max_len = 1  # At least every single char is a palindrome

    for i in range(n):
        # Odd-length palindromes, center at i
        l, r = i, i
        while l >= 0 and r < n and S[l] == S[r]:
            curr_len = r - l + 1
            if curr_len > max_len:
                max_len = curr_len
            l -= 1
            r += 1

        # Even-length palindromes, center between i and i+1
        l, r = i, i + 1
        while l >= 0 and r < n and S[l] == S[r]:
            curr_len = r - l + 1
            if curr_len > max_len:
                max_len = curr_len
            l -= 1
            r += 1

    print(max_len)

if __name__ == "__main__":
    main()