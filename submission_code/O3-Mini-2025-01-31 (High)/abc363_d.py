def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    
    # The sequence of palindrome numbers (in increasing order) starts with 0.
    # So if N == 1, then answer is 0.
    if N == 1:
        sys.stdout.write("0")
        return

    # For positive palindromes (without leading zeros) the next numbers are:
    # 1-digit: 1, 2, …, 9         → count = 9
    # 2-digit: 11,22,...,99        → count = 9   (constructed from a one-digit "half")
    # 3-digit: For half of length 2, from 10 to 99, palindrome = "ab" + "a" (s + s[:-1][::-1])
    # 4-digit: Same half range as 3-digit, palindrome = "ab" + "ba" (s + s[::-1])
    # In general, for a d-digit palindrome with d >= 1 (excluding the 0 already counted),
    # we choose k = ceil(d/2) digits. The first digit must be non-zero.
    # So the number of d-digit positive palindromes is: 9 * (10^(k-1))
    
    # Adjust N to count only the positive palindromes (since index 1 was 0).
    rem_index = N - 1  # 1-indexed position in the positive palindrome list

    # We'll iterate over d = 1,2,3,... until we find the group (digit length) in which
    # the rem_index-th positive palindrome lies.
    d = 1
    while True:
        k = (d + 1) // 2  # length of the "half" we need to choose
        group_count = 9 * pow(10, k - 1)  # number of d-digit palindromes (for d>=1)
        if rem_index > group_count:
            rem_index -= group_count
            d += 1
        else:
            break

    # Now, within the d-digit group, the palindromes are generated in increasing order
    # by taking the "half" number starting at base = 10^(k-1) (for k>=1) and then forming
    # the full palindrome.
    # rem_index is 1-indexed in the current group.
    offset = rem_index - 1  # convert to 0-indexed offset
    base = pow(10, k - 1)
    half = base + offset
    s = str(half)

    # Build the full palindrome based on whether the number of digits d is odd or even.
    if d % 2 == 0:
        # Even number of digits: mirror the whole half.
        palindrome = s + s[::-1]
    else:
        # Odd number of digits: mirror all but the last digit of the half.
        palindrome = s + s[:-1][::-1]
    
    sys.stdout.write(palindrome)
    
if __name__ == '__main__':
    main()