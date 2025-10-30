def check_pattern(W, B):
    # Base pattern of piano keys
    pattern = "wbwbwwbwbwbw"
    # Create a string of 3 patterns to handle any possible substring
    # (2 complete patterns plus some extra to handle edge cases)
    extended = pattern * 3
    
    # Try all possible starting positions in one complete pattern
    for start in range(len(pattern)):
        count_w = 0
        count_b = 0
        # Check if we can find required counts starting from this position
        for i in range(start, len(extended)):
            if count_w > W or count_b > B:
                break
            if extended[i] == 'w':
                count_w += 1
            else:
                count_b += 1
            if count_w == W and count_b == B:
                return True
    return False

# Read input
W, B = map(int, input().split())

# Print result
print("Yes" if check_pattern(W, B) else "No")