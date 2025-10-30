def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return

    # Read N which is 1-indexed (the 1st palindrome is 0)
    N = int(data[0])
    
    # The ordered sequence of palindromes starts with:
    # Group 1 (1-digit numbers): 0, 1, 2, ..., 9 (count = 10)
    # Then group 2 (2-digit palindromes): 11, 22, ..., 99 (count = 9)
    # Then group 3 (3-digit palindromes): e.g. 101, 111, ... (count = 9*10 = 90)
    # ...
    # For groups with digits d>=2:
    #   If d is even: the palindrome is determined by the first half of digits.
    #       left_half length = d/2 and count = 9 * 10^(d/2 - 1)
    #   If d is odd: left_half length = (d+1)//2 (includes the middle digit)
    #       count = 9 * 10^((d - 1) // 2)
    #
    # Note: Since N can be as large as 10^18, we loop finding the group that contains the N-th palindrome.
    
    # Handle group with 1-digit separately.
    if N <= 10:
        # The sequence of 1-digit palindromes in order is: 0,1,2,...,9.
        sys.stdout.write(str(N - 1))
        return

    # We have already covered the 1-digit palindromes.
    cumulative = 10  # total count so far (group 1)
    d = 2  # starting with 2-digit palindromes

    # Iterate until we find the digit group where the cumulative count >= N.
    while True:
        if d % 2 == 0:
            # For even number of digits: left half length = d/2,
            # and count = 9 * 10^(d/2 - 1)
            half_len = d // 2
            group_count = 9 * (10 ** (half_len - 1))
        else:
            # For odd number of digits: left half length = (d + 1) // 2
            # and count = 9 * 10^((d - 1) // 2)
            half_len = (d + 1) // 2
            group_count = 9 * (10 ** ((d - 1) // 2))
        
        if cumulative + group_count >= N:
            break
        cumulative += group_count
        d += 1

    # Find the offset (0-indexed) within the current digit-length group.
    offset = N - cumulative - 1

    # Construct the palindrome.
    # For d-digit palindromes (when d >= 2), the palindrome is constructed from a "left half"
    # that uniquely determines the number.
    if d % 2 == 0:
        # Even number of digits.
        # The left half is of length d/2 and must not start with 0.
        half_len = d // 2
        start = 10 ** (half_len - 1)
        left_half = start + offset
        s = str(left_half)
        palindrome = s + s[::-1]
    else:
        # Odd number of digits.
        # The left half is of length (d + 1) // 2.
        # We mirror all except the last digit (which is the middle one).
        half_len = (d + 1) // 2
        start = 10 ** (half_len - 1)
        left_half = start + offset
        s = str(left_half)
        palindrome = s + s[-2::-1]
    
    sys.stdout.write(str(palindrome))


if __name__ == '__main__':
    main()