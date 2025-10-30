def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    D = int(data[0])
    
    ans = float('inf')
    # x can range from 0 to sqrt(D) + 1 because even when x^2 > D,
    # the candidate (x, 0) can be competitive.
    x_max = math.isqrt(D) + 1

    for x in range(x_max + 1):
        x2 = x * x
        rem = D - x2
        if rem < 0:
            # For x^2 > D, only candidate is y = 0.
            diff = abs(x2 - D)
            if diff < ans:
                ans = diff
            continue

        # The ideal y approximates sqrt(rem)
        y = math.isqrt(rem)
        # Check for both y and y+1 to cover both sides.
        for cand in [y, y + 1]:
            candidate_sum = x2 + cand * cand
            diff = abs(candidate_sum - D)
            if diff < ans:
                ans = diff
                # If found exact match, short-circuit the search.
                if ans == 0:
                    sys.stdout.write("0")
                    return

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()