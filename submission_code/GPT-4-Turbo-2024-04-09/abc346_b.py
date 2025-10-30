def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    W, B = map(int, data.split())
    
    # The repeating pattern in the piano keyboard
    pattern = "wbwbwwbwbwbw"
    pattern_length = len(pattern)
    
    # Count w's and b's in the pattern
    count_w = pattern.count('w')
    count_b = pattern.count('b')
    
    # We need to check if we can find a substring with exactly W 'w's and B 'b's
    # We will use a sliding window approach over the pattern repeated enough times
    # Since the pattern repeats every 12 characters, we need to consider at most
    # two full cycles of the pattern to account for any possible wrap-around substring
    
    # We will use a window of length at most 2 * pattern_length
    extended_pattern = pattern * 2
    
    # To find a substring with exactly W 'w's and B 'b's, we will use a sliding window
    # with a dynamic length and count the occurrences of 'w' and 'b'
    
    # Initialize counts
    current_w = 0
    current_b = 0
    
    # We use a two-pointer technique to expand and contract a window
    left = 0
    
    # We need to consider windows of various lengths, so we iterate over all possible starting points
    for right in range(len(extended_pattern)):
        if extended_pattern[right] == 'w':
            current_w += 1
        elif extended_pattern[right] == 'b':
            current_b += 1
        
        # Now adjust the left pointer to maintain the counts
        while current_w > W or current_b > B:
            if extended_pattern[left] == 'w':
                current_w -= 1
            elif extended_pattern[left] == 'b':
                current_b -= 1
            left += 1
        
        # Check if we have the exact match
        if current_w == W and current_b == B:
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()