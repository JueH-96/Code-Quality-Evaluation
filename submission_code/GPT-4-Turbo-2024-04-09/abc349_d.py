def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    L, R = map(int, data.split())
    
    segments = []
    
    while L < R:
        # Find the largest power of two that fits within the current L
        max_power = 1
        while max_power <= L:
            max_power <<= 1
        
        # If max_power is greater than L, we need to step back one power of two
        if max_power > L:
            max_power >>= 1
        
        # Now, find the largest segment starting from L that is a good sequence
        if max_power == 0:
            max_power = 1
        
        # Find the largest power of two that can define a range starting from L
        start = L
        while (start & (max_power - 1)) != 0:
            max_power >>= 1
        
        # Define the end of the segment
        end = start + max_power
        if end > R:
            end = R
        
        # Append the segment
        segments.append((start, end))
        
        # Move L to end
        L = end
    
    # Output the result
    print(len(segments))
    for l, r in segments:
        print(l, r)

if __name__ == "__main__":
    main()