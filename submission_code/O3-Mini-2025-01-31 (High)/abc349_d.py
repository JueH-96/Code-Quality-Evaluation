def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    L = int(data[0])
    R = int(data[1])
    
    # Explanation:
    # A "good sequence" is defined as S(l, r) = (l, l+1, …, r-1) such that
    # its length (r-l) is a power of 2 (say 2^i) and the left endpoint l is divisible by 2^i.
    # Equivalently, we must have r - l = 2^i and l % (r-l) == 0.
    #
    # It is known that any interval [L, R) can be uniquely partitioned into these dyadic 
    # intervals (good sequences) by “greedily” choosing the largest good segment starting at
    # the current position.
    #
    # For a given starting point x, the candidate segment [x, x+p] is good if:
    #   • p is a power of two,
    #   • x % p == 0, and
    #   • p <= R - x (so that the segment is completely contained in [L, R)).
    #
    # Note that for any nonzero x, the largest power-of-two that divides x is given by (x & -x)
    # (its lowest set bit). This tells us that if x is not zero, then the allowed segment lengths
    # are powers-of-two not exceeding (x & -x).
    # In contrast, if x == 0 then 0 is divisible by every power of 2 so the only restriction
    # is that the segment does not overshoot R.
    #
    # Thus, with "rem" = R - x, let block2 be the largest power-of-two at most rem.
    # Then the candidate segment's length is:
    #    candidate = block2    if x == 0,
    #    candidate = min(x & -x, block2)  if x != 0.
    #
    # We then append the segment [x, x + candidate] and update x to x + candidate.
    
    segments = []
    x = L
    while x < R:
        rem = R - x
        # Compute block2: the largest power-of-two not exceeding rem.
        block2 = 1 << (rem.bit_length() - 1)
        if x == 0:
            candidate = block2
        else:
            candidate = min(x & -x, block2)
        segments.append((x, x + candidate))
        x += candidate

    # Output the number of segments and then each segment
    out_lines = []
    out_lines.append(str(len(segments)))
    for l, r in segments:
        out_lines.append(f"{l} {r}")
    sys.stdout.write("
".join(out_lines))


if __name__ == '__main__':
    main()