def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    
    # We'll keep track of the maximum contiguous block length for each character.
    # There are 26 letters, so we use a list of length 26 initialized to 0.
    max_block = [0]*26
    
    current_char = None
    current_length = 0
    
    for ch in S:
        idx = ord(ch) - ord('a')
        if ch == current_char:
            # Continue the current block
            current_length += 1
        else:
            # Start a new block
            current_char = ch
            current_length = 1
        # Update max block length for this character
        if current_length > max_block[idx]:
            max_block[idx] = current_length
    
    # The number of distinct substrings that are repetitions of one character
    # is the sum over all characters of (1 up to max_block[char]), i.e. max_block[char].
    print(sum(max_block))

# Let's call solve()
# solve()