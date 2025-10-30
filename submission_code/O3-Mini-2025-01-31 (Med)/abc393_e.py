def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # determine the maximum value in A
    max_val = max(A)
    
    # Build a frequency table for all numbers up to max_val.
    freq = [0] * (max_val + 1)
    for a in A:
        freq[a] += 1

    # For each d from 1 to max_val, count how many numbers in A are divisible by d.
    # This step runs in roughly sum_{d=1}^{max_val} (max_val/d) iterations.
    count = [0] * (max_val + 1)
    for d in range(1, max_val + 1):
        s = 0
        for m in range(d, max_val + 1, d):
            s += freq[m]
        count[d] = s

    # Precompute a lookup for every number m (1 <= m <= max_val) giving the maximum divisor d
    # such that d divides m and count[d] >= k.
    # We do this by looping d from max_val down to 1. For every multiple m of d,
    # if we have not already assigned an answer, then d is the maximum valid divisor for m.
    ans_for = [0] * (max_val + 1)
    for d in range(max_val, 0, -1):
        if count[d] >= k:
            for m in range(d, max_val + 1, d):
                if ans_for[m] == 0:
                    ans_for[m] = d
    
    # For each element A[i] of the input sequence, output the answer.
    out_lines = []
    for a in A:
        out_lines.append(str(ans_for[a]))
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()