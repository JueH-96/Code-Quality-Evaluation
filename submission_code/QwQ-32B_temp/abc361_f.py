import bisect
import math

def main():
    N = int(input().strip())
    if N < 1:
        print(0)
        return
    total = 1  # Count 1

    # Compute L = floor(sqrt(N))
    L = int(N ** 0.5)
    while (L + 1) * (L + 1) <= N:
        L += 1
    while L * L > N:
        L -= 1

    # Generate all perfect powers up to L
    perfect_pows = set()
    if L >= 2:
        max_k = int(math.log2(L)) + 2  # To cover possible exponents
        for k in range(2, max_k + 1):
            m = 2
            while True:
                val = m ** k
                if val > L:
                    break
                perfect_pows.add(val)
                m += 1
    perfect_pows = sorted(perfect_pows)

    # Determine max_e
    max_e = 1
    if N >= 2:
        max_e = int(math.log2(N)) + 2

    for e in range(2, max_e + 1):
        # Compute a_max via binary search
        low = 2
        high = N
        best = 1
        while low <= high:
            mid = (low + high) // 2
            product = 1
            over = False
            for _ in range(e):
                product *= mid
                if product > N:
                    over = True
                    break
            if over:
                high = mid - 1
            else:
                best = mid
                low = mid + 1
        a_max = best
        if a_max < 2:
            continue
        total_a = a_max - 1
        # Count how many perfect_pows are <= a_max
        cnt = bisect.bisect_right(perfect_pows, a_max)
        non_powers = total_a - cnt
        total += non_powers

    print(total)

if __name__ == "__main__":
    main()