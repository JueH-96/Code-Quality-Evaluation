import sys
import threading

def main():
    import sys, math

    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    S = data[1].strip()

    # Count non-zero digits and zeros in S
    S_count9 = [0]*10  # we'll use indices 1..9
    zCount = 0
    for ch in S:
        d = ord(ch) - 48
        if d == 0:
            zCount += 1
        else:
            S_count9[d] += 1

    # M = total non-zero digits required in a square
    M = sum(S_count9)
    # If all digits are zero, only "0" is possible
    if M == 0:
        print(1)
        return

    # Digit‐length constraints for the square itself (without padding zeros).
    # Let L_low = min digits in m = max(1, N - zeros_in_S)
    #       L_high = max digits in m = N
    L_low = N - zCount
    if L_low < 1:
        L_low = 1
    L_high = N

    # Compute bounds for k = sqrt(m)
    # m must be in [10^(L_low-1), 10^L_high - 1]
    # so k in [ ceil(sqrt(10^(L_low-1))), floor(sqrt(10^L_high - 1)) ]
    low_thresh = 10**(L_low - 1)
    high_thresh = 10**L_high - 1

    k0 = math.isqrt(low_thresh)
    if k0*k0 < low_thresh:
        k_low = k0 + 1
    else:
        k_low = k0
    k_high = math.isqrt(high_thresh)

    ans = 0

    # We'll count only non-zero digits of m, matching S_count9 exactly.
    # Reuse a small array counts9 to track counts of digits 1..9 in m.
    counts9 = [0]*10
    # used_digits will record which non-zero digits we touched in the last iteration,
    # so we can reset counts9[d] = 0 for those d before the next iteration.
    used_digits = []

    for k in range(k_low, k_high + 1):
        m = k*k
        # Quick filter on last digit: it must appear in S
        d_last = m % 10
        if d_last == 0:
            if zCount == 0:
                continue
        else:
            if S_count9[d_last] == 0:
                continue

        # Reset counts9 for digits used in previous square
        for d in used_digits:
            counts9[d] = 0
        used_digits.clear()

        # Scan the decimal digits of m, counting only non-zero digits.
        # Break early if any digit count would exceed S_count9[d].
        tmp = m
        scan_sum = 0
        while tmp:
            d = tmp % 10
            tmp //= 10
            if d != 0:
                c = counts9[d] + 1
                if c > S_count9[d]:
                    break
                counts9[d] = c
                scan_sum += 1
                if c == 1:
                    used_digits.append(d)
        else:
            # If we did not break, we scanned all digits of m.
            # If total non-zero digits equals M, then the multiset of non-zero digits
            # matches exactly; zeros (both in m and padded) necessarily match S.
            if scan_sum == M:
                ans += 1

    print(ans)

if __name__ == "__main__":
    main()