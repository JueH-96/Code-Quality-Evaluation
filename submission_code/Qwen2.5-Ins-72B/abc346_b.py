# YOUR CODE HERE
W, B = map(int, input().split())

# The repeating pattern of the infinite keyboard
pattern = "wbwbwwbwbwbw"

# Function to check if a substring with W whites and B blacks exists
def can_form_substring(W, B):
    # Iterate over all possible starting positions in the pattern
    for start in range(len(pattern)):
        white_count = 0
        black_count = 0
        # Check the next 12 characters (one full cycle of the pattern)
        for i in range(start, start + 12):
            if pattern[i % len(pattern)] == 'w':
                white_count += 1
            else:
                black_count += 1
            # If we find a valid substring, return True
            if white_count == W and black_count == B:
                return True
    return False

# Output the result
if can_form_substring(W, B):
    print("Yes")
else:
    print("No")