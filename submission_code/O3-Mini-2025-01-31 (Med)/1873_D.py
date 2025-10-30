def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    pos = 1
    answers = []
    
    # We want to cover all black cell indices using operations that convert
    # any chosen k consecutive cells to white.
    #
    # Notice that if we decide to cover a black cell at position b, we can choose
    # any segment with start L such that L <= b <= L+k-1. To maximize covering future
    # black cells, we want the segment to stretch as far right as possible.
    # In fact, if b <= n-k then we can choose L = b (which yields segment covering b to b+k-1)
    # and if b > n-k then only L=n-k is available (segment covering from n-k to n-1).
    # Then we greedily cover all black cells that fall in that segment.
    
    for _ in range(t):
        n = int(data[pos]); pos += 1
        k = int(data[pos]); pos += 1
        s = data[pos]; pos += 1
        
        # Collect indices of black cells.
        blacks = [i for i, ch in enumerate(s) if ch == 'B']
        
        if not blacks:
            answers.append("0")
            continue
        
        count = 0
        i = 0
        while i < len(blacks):
            b = blacks[i]
            # For b, choose a segment that goes as far right as possible.
            if b <= n - k:
                cover_end = b + k - 1
            else:
                cover_end = n - 1
            # Count all blacks within [b, cover_end]
            while i < len(blacks) and blacks[i] <= cover_end:
                i += 1
            count += 1
        
        answers.append(str(count))
    
    sys.stdout.write("
".join(answers))

if __name__ == '__main__':
    main()