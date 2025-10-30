def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    s = list(next(it))
    
    # Precompute the initial count of "ABC" occurrences in s.
    count_occ = 0
    for i in range(n - 2):
        if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
            count_occ += 1

    out_lines = []
    
    # Process queries one by one.
    for _ in range(q):
        pos = int(next(it)) - 1  # Convert to 0-index.
        c = next(it)
        # If there is no change, output the current count.
        if s[pos] == c:
            out_lines.append(str(count_occ))
            continue
        
        # The update at position pos affects occurrences starting at indices:
        # pos-2, pos-1, and pos (if they are valid positions).
        for j in (pos - 2, pos - 1, pos):
            if j >= 0 and j + 2 < n:
                if s[j] == 'A' and s[j+1] == 'B' and s[j+2] == 'C':
                    count_occ -= 1
        
        # Perform the update.
        s[pos] = c
        
        for j in (pos - 2, pos - 1, pos):
            if j >= 0 and j + 2 < n:
                if s[j] == 'A' and s[j+1] == 'B' and s[j+2] == 'C':
                    count_occ += 1
        
        out_lines.append(str(count_occ))
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()