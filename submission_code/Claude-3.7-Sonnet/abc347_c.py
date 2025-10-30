def solve():
    N, A, B = map(int, input().split())
    D = list(map(int, input().split()))
    
    AB = A + B
    
    # Initialize the valid ranges
    valid_ranges = []
    
    for d in D:
        r = d % AB
        
        # Compute the valid ranges for the current d
        current_ranges = []
        
        if A - r >= 1:
            current_ranges.append((1, A - r))
        
        if r >= 1:
            current_ranges.append((AB - r + 1, AB))
        
        # If there are no valid ranges for the current d, return "No"
        if not current_ranges:
            return "No"
        
        # Intersect the current ranges with the existing valid ranges
        if not valid_ranges:
            valid_ranges = current_ranges
        else:
            new_ranges = []
            for start1, end1 in valid_ranges:
                for start2, end2 in current_ranges:
                    start = max(start1, start2)
                    end = min(end1, end2)
                    if start <= end:
                        new_ranges.append((start, end))
            
            valid_ranges = new_ranges
            
            # If there are no valid ranges after intersection, return "No"
            if not valid_ranges:
                return "No"
    
    return "Yes"

print(solve())