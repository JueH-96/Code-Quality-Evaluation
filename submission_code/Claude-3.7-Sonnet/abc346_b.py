def solve_piano_keyboard(W, B):
    # The infinitely repeating pattern
    pattern = "wbwbwwbwbwbw"
    pattern_length = len(pattern)
    total_length = W + B
    
    if total_length == 0:
        return "Yes"  # Empty substring always exists
    
    # Create an extended pattern to handle wraparound cases
    # We need enough repetitions to cover all possible starting positions
    repetitions_needed = (total_length + pattern_length - 1) // pattern_length + 1
    extended_pattern = pattern * repetitions_needed
    
    # Check all possible starting positions within one pattern cycle
    for i in range(pattern_length):
        substring = extended_pattern[i:i+total_length]
        w_count = substring.count('w')
        b_count = substring.count('b')
        
        if w_count == W and b_count == B:
            return "Yes"
    
    return "No"

# Read input
W, B = map(int, input().split())
print(solve_piano_keyboard(W, B))