def solve():
    N, M, A, B = map(int, input().split())
    bad_ranges = []
    for _ in range(M):
        L, R = map(int, input().split())
        bad_ranges.append((L, R))
    
    def is_bad(pos):
        for L, R in bad_ranges:
            if L <= pos <= R:
                return True
        return False
    
    # Maintain intervals of reachable positions
    reachable = [(1, 1)]
    
    while reachable:
        new_reachable = []
        
        for start, end in reachable:
            # From positions in [start, end], we can reach [start+A, end+B]
            new_start = start + A
            new_end = min(end + B, N)
            
            if new_start > N:
                continue
                
            # Check if we can reach N
            if new_end >= N and not is_bad(N):
                return "Yes"
            
            # Split [new_start, new_end] by removing bad positions
            current = new_start
            while current <= new_end:
                if is_bad(current):
                    current += 1
                else:
                    # Find the end of this good segment
                    segment_end = current
                    while segment_end + 1 <= new_end and not is_bad(segment_end + 1):
                        segment_end += 1
                    new_reachable.append((current, segment_end))
                    current = segment_end + 1
        
        # Merge overlapping intervals
        if new_reachable:
            new_reachable.sort()
            merged = [new_reachable[0]]
            for start, end in new_reachable[1:]:
                if start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            
            # Check if N is in any interval
            for start, end in merged:
                if start <= N <= end:
                    return "Yes"
            
            reachable = merged
        else:
            break
    
    return "No"

print(solve())