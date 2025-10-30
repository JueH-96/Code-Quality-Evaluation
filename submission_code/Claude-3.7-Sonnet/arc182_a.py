def main():
    N, Q = map(int, input().split())
    operations = []
    for _ in range(Q):
        P, V = map(int, input().split())
        operations.append((P, V))
    
    MOD = 998244353
    memo = {}
    
    def update_segments_prefix(segments, P, V):
        new_segments = []
        for start, end, value in segments:
            if end <= P:
                # Segment completely within prefix
                new_segments.append((start, end, V))
            elif start <= P < end:
                # Segment intersects with the prefix
                new_segments.append((start, P, V))
                new_segments.append((P + 1, end, value))
            else:
                # Segment completely outside prefix
                new_segments.append((start, end, value))
        
        # Merge adjacent segments with the same value
        merged_segments = []
        for start, end, value in sorted(new_segments):
            if merged_segments and merged_segments[-1][1] + 1 == start and merged_segments[-1][2] == value:
                merged_segments[-1] = (merged_segments[-1][0], end, value)
            else:
                merged_segments.append((start, end, value))
        
        return tuple(merged_segments)
    
    def update_segments_suffix(segments, P, V):
        new_segments = []
        for start, end, value in segments:
            if end < P:
                # Segment completely outside suffix
                new_segments.append((start, end, value))
            elif start < P <= end:
                # Segment intersects with the suffix
                new_segments.append((start, P - 1, value))
                new_segments.append((P, end, V))
            else:
                # Segment completely within suffix
                new_segments.append((start, end, V))
        
        # Merge adjacent segments with the same value
        merged_segments = []
        for start, end, value in sorted(new_segments):
            if merged_segments and merged_segments[-1][1] + 1 == start and merged_segments[-1][2] == value:
                merged_segments[-1] = (merged_segments[-1][0], end, value)
            else:
                merged_segments.append((start, end, value))
        
        return tuple(merged_segments)
    
    def count_valid_sequences(i, segments):
        if i >= Q:
            return 1  # Found a valid sequence
        
        if (i, segments) in memo:
            return memo[(i, segments)]
        
        P, V = operations[i]
        valid_sequences = 0
        
        # Check if prefix operation would make Snuke cry
        can_do_prefix = True
        for start, end, value in segments:
            if start <= P and value > V:
                can_do_prefix = False
                break
        
        if can_do_prefix:
            # Update segments for prefix operation
            new_segments = update_segments_prefix(segments, P, V)
            valid_sequences = (valid_sequences + count_valid_sequences(i + 1, new_segments)) % MOD
        
        # Check if suffix operation would make Snuke cry
        can_do_suffix = True
        for start, end, value in segments:
            if end >= P and value > V:
                can_do_suffix = False
                break
        
        if can_do_suffix:
            # Update segments for suffix operation
            new_segments = update_segments_suffix(segments, P, V)
            valid_sequences = (valid_sequences + count_valid_sequences(i + 1, new_segments)) % MOD
        
        memo[(i, segments)] = valid_sequences
        return valid_sequences
    
    # Initial state: one segment from 1 to N with value 0
    initial_segments = ((1, N, 0),)
    result = count_valid_sequences(0, initial_segments)
    print(result)

if __name__ == "__main__":
    main()