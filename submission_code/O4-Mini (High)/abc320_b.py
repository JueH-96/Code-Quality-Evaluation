def main():
    import sys
    S = sys.stdin.readline().strip()
    n = len(S)
    max_len = 1  # at least one-character palindromes always exist

    for center in range(n):
        # odd-length palindromes, center is at 'center'
        left = right = center
        while left >= 0 and right < n and S[left] == S[right]:
            curr_len = right - left + 1
            if curr_len > max_len:
                max_len = curr_len
            left -= 1
            right += 1

        # even-length palindromes, center between 'center' and 'center+1'
        left, right = center, center + 1
        while left >= 0 and right < n and S[left] == S[right]:
            curr_len = right - left + 1
            if curr_len > max_len:
                max_len = curr_len
            left -= 1
            right += 1

    print(max_len)

if __name__ == "__main__":
    main()