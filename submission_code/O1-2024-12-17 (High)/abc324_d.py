def main():
    import sys
    import math

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Count digit frequencies of S and sum of digits.
    freq_s = [0]*10
    sum_s = 0
    for ch in S:
        d = ord(ch) - ord('0')
        freq_s[d] += 1
        sum_s += d

    # Special case: all digits are zero.
    if freq_s[0] == N:
        # Only the number 0 can be formed, which is 0^2.
        print(1)
        return

    freq_s = tuple(freq_s)  # Make it a tuple for fast comparison.
    mod9 = sum_s % 9

    # Squares mod 9 can only be 0,1,4,7.
    # Each residue in {0,1,4,7} comes from i mod 9 in specific sets:
    # 0 -> i mod 9 in {0,3,6}
    # 1 -> i mod 9 in {1,8}
    # 4 -> i mod 9 in {2,7}
    # 7 -> i mod 9 in {4,5}
    squares_mod_9 = {
        0: [0, 3, 6],
        1: [1, 8],
        4: [2, 7],
        7: [4, 5]
    }

    if mod9 not in squares_mod_9:
        # If sum of digits mod 9 is not in {0,1,4,7}, no square can match.
        print(0)
        return

    # Upper bound for the values we can form: < 10^N
    max_val = 10**N
    i_max = math.isqrt(max_val - 1)

    # We also know that if we need exactly N digits (including leading zeros),
    # we cannot pad more zeros than freq_s[0]. So i^2 must have at least
    # (N - freq_s[0]) digits if freq_s[0] < N.
    # d >= N - freq_s[0]. So i^2 >= 10^( (N - freq_s[0]) - 1 ).
    # If that exponent is < 0, we handled it by the all-zero case above.
    min_digits = N - freq_s[0]  # This is > 0, otherwise handled above.
    min_val = 10**(min_digits - 1) if min_digits > 0 else 1
    # i^2 >= min_val => i >= sqrt(min_val).
    # We'll pick i_min to be the smallest integer whose square >= min_val.
    if min_val > 1:
        i_min = math.isqrt(min_val - 1) + 1
    else:
        i_min = 0  # This allows checking i=0 or i=1, etc.

    count_squares = 0
    for r in squares_mod_9[mod9]:
        # We only iterate i where i mod 9 = r, starting from at least i_min.
        if i_min > 0:
            remainder = i_min % 9
            diff = (r - remainder) % 9
            start = i_min + diff
        else:
            # If i_min = 0, we can start at r directly.
            start = r

        i = start
        while i <= i_max:
            sq = i*i
            if sq >= max_val:
                break

            # Build digit frequency of i^2 (then pad leading zeros).
            sq_str = str(sq)
            d = len(sq_str)
            freq = [0]*10
            for c in sq_str:
                freq[ord(c) - ord('0')] += 1
            freq[0] += (N - d)

            if tuple(freq) == freq_s:
                count_squares += 1

            i += 9

    print(count_squares)

# Do not forget to call main().
main()