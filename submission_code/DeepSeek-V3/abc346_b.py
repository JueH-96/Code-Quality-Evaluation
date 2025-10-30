# YOUR CODE HERE
def can_form_substring(W, B):
    # The repeating pattern of the piano keyboard
    pattern = "wbwbwwbwbwbw"
    pattern_length = len(pattern)
    
    # Precompute the cumulative counts of 'w' and 'b' in the pattern
    w_counts = [0] * (pattern_length + 1)
    b_counts = [0] * (pattern_length + 1)
    
    for i in range(pattern_length):
        w_counts[i+1] = w_counts[i] + (1 if pattern[i] == 'w' else 0)
        b_counts[i+1] = b_counts[i] + (1 if pattern[i] == 'b' else 0)
    
    # Since the pattern repeats, we can consider the pattern multiple times
    # to cover all possible substrings of length up to W + B
    # We need to check all possible starting positions in the pattern
    # and see if any substring of length (W + B) has exactly W 'w's and B 'b's
    
    # The maximum length we need to consider is the length of the pattern multiplied by the number of repetitions
    # Since W and B are up to 100, and the pattern length is 12, we can limit the number of repetitions
    # to a reasonable number, say 10, to cover all possible combinations
    
    max_repeats = 10
    total_length = pattern_length * max_repeats
    
    # Precompute the cumulative counts for the extended pattern
    extended_w_counts = [0] * (total_length + 1)
    extended_b_counts = [0] * (total_length + 1)
    
    for i in range(total_length):
        extended_w_counts[i+1] = extended_w_counts[i] + (1 if pattern[i % pattern_length] == 'w' else 0)
        extended_b_counts[i+1] = extended_b_counts[i] + (1 if pattern[i % pattern_length] == 'b' else 0)
    
    # Now, for each possible starting position, check all possible lengths
    # Since the pattern repeats, we can limit the starting positions to the first pattern_length
    for start in range(pattern_length):
        # We need to find a substring starting at start and ending at some end
        # such that the number of 'w's is W and the number of 'b's is B
        # The length of the substring is (W + B)
        length = W + B
        if length == 0:
            continue
        # Calculate the end position
        end = start + length
        if end > total_length:
            continue
        # Get the counts of 'w' and 'b' in this substring
        current_w = extended_w_counts[end] - extended_w_counts[start]
        current_b = extended_b_counts[end] - extended_b_counts[start]
        if current_w == W and current_b == B:
            return True
    return False

# Read input
W, B = map(int, input().split())

# Check if such a substring exists
if can_form_substring(W, B):
    print("Yes")
else:
    print("No")