def main():
    import sys
    import math

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Frequency of digits in S
    freqS = [0]*10
    sumDigit = 0
    for ch in S:
        d = ord(ch) - ord('0')
        freqS[d] += 1
        sumDigit += d

    # Squares in base 10 can only have digit-sum mod 9 in {0,1,4,7}.
    sd = sumDigit % 9
    if sd not in (0, 1, 4, 7):
        print(0)
        return

    # Determine which residues mod 9 produce square ≡ sd (mod 9).
    # x^2 mod 9 can be 0,1,4,7, so we collect x if x^2 % 9 == sd.
    squares_mod9 = [x for x in range(9) if (x*x) % 9 == sd]
    if not squares_mod9:
        print(0)
        return

    # Largest integer we can form has N digits of '9' => 10^N - 1
    largest = 10**N - 1
    max_root = int(math.isqrt(largest))

    count = 0
    freqS_tuple = tuple(freqS)

    for i in range(max_root + 1):
        # Skip if i^2 mod 9 != sd
        if (i % 9) not in squares_mod9:
            continue

        sq = i * i
        # Quick check: skip if the last digit is impossible
        last_dig = sq % 10
        if freqS[last_dig] == 0:
            continue

        # If sq has more than N digits, break (i is large enough)
        sq_str = str(sq)
        if len(sq_str) > N:
            break

        # Build frequency of digits for sq (with leading zeros if needed)
        freq = [0]*10
        needed_zeros = N - len(sq_str)
        freq[0] = needed_zeros
        for c in sq_str:
            freq[ord(c) - ord('0')] += 1

        # Compare frequency
        if tuple(freq) == freqS_tuple:
            count += 1

    print(count)

# Do not forget to call main()
if __name__ == "__main__":
    main()