def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    A.sort()
    A_N = A[-1]
    # Compute M = 2^100 - A_N - 1
    # Since 2^100 is a large number, it's better to use left shift
    power_100 = 1 << 100
    M = power_100 - A_N - 1
    
    events = []
    for a in A:
        # Generate events for m from 1 to 300, stepping by 2 (odd m)
        for m in range(1, 301, 2):
            lower = 1 << (m - 1)
            upper = 1 << m
            s = lower - a
            e = upper - a
            # Clamp to [1, M + 1)
            s_clamped = max(s, 1)
            e_clamped = min(e, M + 1)
            if s_clamped < e_clamped:
                events.append((s_clamped, 1))
                events.append((e_clamped, -1))
    
    # Sort events by position and delta (end events come first)
    events.sort()
    
    max_coverage = 0
    current_coverage = 0
    prev_pos = None
    
    for pos, delta in events:
        if prev_pos is not None:
            start = max(prev_pos, 1)
            end = min(pos, M + 1)
            if start < end:
                # The interval (start, end) is covered by current_coverage
                if current_coverage > max_coverage:
                    max_coverage = current_coverage
        
        current_coverage += delta
        prev_pos = pos
    
    # Process the remaining interval after the last event
    if prev_pos is not None:
        start = max(prev_pos, 1)
        end = M + 1
        if start < end:
            if current_coverage > max_coverage:
                max_coverage = current_coverage
    
    print(max_coverage)

if __name__ == "__main__":
    main()