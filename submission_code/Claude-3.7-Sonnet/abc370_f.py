def all_possible_valid_divisions(pieces, K, min_mass):
    """
    Find all possible divisions of the cake into K segments
    such that each segment has a mass of at least min_mass.
    
    Returns a list of divisions, where each division is a list of segments.
    A segment is a tuple (start, end) of indices of the first and last piece in the segment.
    """
    n = len(pieces)
    valid_divisions = []
    
    for start in range(n):
        segments = []
        current_mass = 0
        segment_start = start
        
        for i in range(n):
            idx = (start + i) % n
            current_mass += pieces[idx]
            
            if current_mass >= min_mass and len(segments) < K - 1:
                segments.append((segment_start, idx))
                current_mass = 0
                segment_start = (idx + 1) % n
        
        # Check if the last segment is valid
        if len(segments) == K - 1 and current_mass >= min_mass:
            segments.append((segment_start, (start + n - 1) % n))
            valid_divisions.append(segments)
    
    return valid_divisions

def find_max_min_mass_and_never_cut_lines(pieces, K):
    """
    Binary search to find the maximum value of the minimum mass
    and the number of cut lines that are never cut.
    """
    n = len(pieces)
    
    # Binary search for the maximum value of the minimum mass
    low = 1
    high = sum(pieces)
    
    while low < high:
        mid = (low + high + 1) // 2
        divisions = all_possible_valid_divisions(pieces, K, mid)
        
        if divisions:
            low = mid
        else:
            high = mid - 1
    
    best_min_mass = low
    optimal_divisions = all_possible_valid_divisions(pieces, K, best_min_mass)
    
    # Identify all cut lines that are cut in any optimal division
    all_cut_lines = set()
    
    for division in optimal_divisions:
        for segment in division:
            _, end = segment
            # The cut line after the end piece of a segment is cut
            all_cut_lines.add(end)
    
    # The number of cut lines that are never cut is n - (number of cut lines that are ever cut)
    never_cut = n - len(all_cut_lines)
    
    return best_min_mass, never_cut

# Read input, solve the problem and output the result
N, K = map(int, input().split())
pieces = list(map(int, input().split()))
best_min_mass, never_cut = find_max_min_mass_and_never_cut_lines(pieces, K)
print(best_min_mass, never_cut)