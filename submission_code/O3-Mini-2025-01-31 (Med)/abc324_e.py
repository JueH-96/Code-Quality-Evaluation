def main():
    import sys
    data = sys.stdin.buffer.read().splitlines()
    if not data:
        return
    # Process the first line: N and T.
    first_line = data[0].split()
    if len(first_line) == 1:
        N = int(first_line[0])
        T = data[1].decode()
        strings = [line.decode() for line in data[2:2+N]]
    else:
        N = int(first_line[0])
        T = first_line[1].decode() if isinstance(first_line[1], bytes) else first_line[1]
        strings = [line.decode() for line in data[1:1+N]]
        
    m = len(T)
    
    # Prepare frequency arrays for p(S) and b(S) where 0 <= value <= m.
    count_p = [0] * (m + 1)
    count_b = [0] * (m + 1)
    
    # For each string, compute p(S)
    for s in strings:
        p = 0  # number of characters matched from T starting at index 0.
        for ch in s:
            if p < m and ch == T[p]:
                p += 1
                if p == m:
                    break
        count_p[p] += 1
        
        # Compute b(S): length of longest suffix of T that can be matched in s.
        # We do so by scanning s in reverse and matching T from its end backward.
        b = 0
        idx = m - 1  # pointer in T (starting from the last character)
        for ch in reversed(s):
            if idx >= 0 and ch == T[idx]:
                b += 1
                idx -= 1
                if idx < 0:
                    break
        count_b[b] += 1

    # Build a cumulative frequency array for count_b:
    # cum_b[x] = number of strings with b(S) >= x.
    cum_b = [0] * (m + 2)  # length m+2 to avoid index issues.
    for b_val in range(m, -1, -1):
        cum_b[b_val] = cum_b[b_val+1] + count_b[b_val]
        
    # Count valid pairs: for each string Sᵢ having p = p_val, 
    # we need Sⱼ with b >= (m - p_val) so that p_val + b >= m.
    total = 0
    for p_val in range(m+1):
        required = m - p_val  # b must be >= required.
        total += count_p[p_val] * cum_b[required]
        
    sys.stdout.write(str(total))

if __name__ == '__main__':
    main()