def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    
    # We count palindromic numbers in increasing order of digits.
    # For L = 1, allowed numbers are 0 to 9 (10 numbers).
    # For L >= 2, palindromes are built by choosing the first ceil(L/2)
    # digits. The first digit (of the half) cannot be 0.
    # For even L, count = 9 * 10^(L/2 - 1)
    # For odd L, count = 9 * 10^((L-1)//2)
    
    L = 1
    while True:
        if L == 1:
            count = 10
        else:
            half_len = (L + 1) // 2
            count = 9 * (10 ** (half_len - 1))
        if N > count:
            N -= count
            L += 1
        else:
            break  # Found the digit-length block containing the N-th palindromic number

    # Now N is within the palindromes with L digits.
    # For the block, use 0-indexed index.
    index = N - 1

    if L == 1:
        # For single-digit palindromes, answer is simply the index (0 for first, 1, 2,...)
        print(index)
        return

    half_len = (L + 1) // 2
    # For L>=2, the smallest candidate half is 10^(half_len-1) (it cannot start with 0)
    start = 10 ** (half_len - 1)
    half = start + index
    half_str = str(half)
    
    # Build the full palindrome from the half.
    if L % 2 == 0:
        # Even length: mirror the entire half.
        palindrome = half_str + half_str[::-1]
    else:
        # Odd length: mirror all except the middle digit.
        palindrome = half_str + half_str[-2::-1]

    print(palindrome)

if __name__ == '__main__':
    main()