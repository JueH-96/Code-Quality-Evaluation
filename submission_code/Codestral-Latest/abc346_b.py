# YOUR CODE HERE
W, B = map(int, input().split())

# The repeating pattern of the keyboard
pattern = "wbwbwwbwbwbw"
pattern_length = len(pattern)

# Function to check if the substring exists
def check_substring(W, B):
    # Calculate the total length of the substring we need to check
    total_length = W + B

    # Check all possible starting positions in the pattern
    for start in range(pattern_length):
        w_count = 0
        b_count = 0
        for i in range(start, start + total_length):
            # Use modulo to wrap around the pattern
            index = i % pattern_length
            if pattern[index] == 'w':
                w_count += 1
            else:
                b_count += 1
            # Early exit if we exceed the required counts
            if w_count > W or b_count > B:
                break
        # Check if we found the required counts
        if w_count == W and b_count == B:
            return True
    return False

# Output the result
if check_substring(W, B):
    print("Yes")
else:
    print("No")