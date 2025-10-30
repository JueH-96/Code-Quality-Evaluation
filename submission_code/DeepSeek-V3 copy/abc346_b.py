# YOUR CODE HERE
def find_substring(W, B):
    # The repeating pattern of the piano keyboard
    pattern = "wbwbwwbwbwbw"
    pattern_length = len(pattern)
    # To handle the infinite nature, we can consider a sufficiently large window
    # Since the pattern repeats every 12 characters, we can consider a window of size (W + B) * 2
    # to ensure we cover all possible starting points within the pattern
    window_size = W + B
    # We will check all possible starting positions within one pattern length
    for start in range(pattern_length):
        w_count = 0
        b_count = 0
        # Iterate through the window starting at 'start'
        for i in range(window_size):
            # Calculate the actual position in the infinite string
            pos = (start + i) % pattern_length
            if pattern[pos] == 'w':
                w_count += 1
            else:
                b_count += 1
            # If we have the required counts, return True
            if w_count == W and b_count == B:
                return True
            # If we exceed the required counts, break and try the next starting position
            if w_count > W or b_count > B:
                break
    return False

# Read input
W, B = map(int, input().split())

# Check if such a substring exists
if find_substring(W, B):
    print("Yes")
else:
    print("No")