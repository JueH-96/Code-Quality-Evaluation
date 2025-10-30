def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    # The palindromes in increasing order start:
    # N=1 -> 0, N=2 -> 1, ..., N=10 -> 9
    # Then length 2,3,4,...
    if N <= 10:
        # 0 through 9
        print(N - 1)
        return

    # Subtract the first 10 palindromes of length 1
    N -= 10

    # Now find the length L of the palindrome group where the N-th lies
    # For L>=2, count(L) = 9 * 10^((L-1)//2)
    L = 1
    while True:
        L += 1
        # number of palindromes of length L
        half_len = (L - 1) // 2  # exponent
        cnt = 9 * pow(10, half_len)
        if N > cnt:
            N -= cnt
        else:
            # the N-th in this length
            idx = N - 1   # zero-based
            # build the first half
            HL = (L + 1) // 2  # number of digits in the "left half"
            start = pow(10, HL - 1)
            first_half_num = start + idx
            s = str(first_half_num)
            if L % 2 == 0:
                # even length: reflect entire s
                pal = s + s[::-1]
            else:
                # odd length: reflect s without the last char
                pal = s + s[-2::-1]
            print(pal)
            return

if __name__ == "__main__":
    main()