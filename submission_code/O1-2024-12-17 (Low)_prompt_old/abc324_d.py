def solve():
    import sys
    import math

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Special case: if all digits are '0', the only number we can form is 0 (which is a square).
    if all(ch == '0' for ch in S):
        print(1)
        return

    # Build frequency array of digits in S
    freqS = [0]*10
    for ch in S:
        freqS[ord(ch) - ord('0')] += 1

    # We'll use a quick mod-9 filter to skip if sum of digits mod 9 is impossible for a square.
    # Squares mod 9 can only be 0,1,4,7.
    sum_digits = sum((i * freqS[i]) for i in range(10))
    sum_digits_mod9 = sum_digits % 9
    valid_sq_mod9 = {0,1,4,7}
    if sum_digits_mod9 not in valid_sq_mod9:
        print(0)
        return

    limit = 10**N  # We only consider squares < 10^N (i.e., up to N digits)
    sqrt_limit = int(math.isqrt(limit - 1))
    ans = 0

    # Check squares from 1^2 up to sqrt_limit^2
    for i in range(sqrt_limit + 1):
        # i^2 mod 9 must match sum_digits_mod9 (quick check)
        if (i % 9) * (i % 9) % 9 != sum_digits_mod9:
            continue

        sq = i*i
        if sq >= limit:  # too large
            break

        # Convert sq to string with zero-padding to length N
        sq_str = str(sq)
        length_sq = len(sq_str)
        if length_sq > N:
            break  # no need to continue if we already exceed digit length

        # Build frequency array for this square
        freq_sq = [0]*10
        # Count leading zeros (if any) to make it length N
        leading_zeros = N - length_sq
        freq_sq[0] = leading_zeros
        for ch in sq_str:
            freq_sq[ord(ch) - ord('0')] += 1

        # Compare frequency arrays
        if freq_sq == freqS:
            ans += 1

    print(ans)

# Let's call solve() to complete
# (In the real submission environment, only solve() would be called externally.)
def main():
    solve()

if __name__ == "__main__":
    main()