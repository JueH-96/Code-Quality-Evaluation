def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    W = int(data[0])
    B = int(data[1])
    L = W + B  # The length of the required substring (since every key is either 'w' or 'b')
    
    # The given pattern repeats infinitely
    pattern = "wbwbwwbwbwbw"   # 12 characters: 7 white ('w') and 5 black ('b')
    # To cover any substring of length L appearing in S (the infinite repetition),
    # we can generate a string long enough by repeating the pattern sufficiently.
    # A safe choice: repeat pattern ((L // len(pattern)) + 2) times.
    rep = (L // len(pattern)) + 2
    s = pattern * rep
    n = len(s)
    
    # We need to find a contiguous substring of s of length L with exactly W w's.
    # (Because if the substring is of length L and has W w's, it automatically has B b's.)
    # Use a sliding window technique.
    
    # Compute count for the first window.
    current_w = s[:L].count('w')
    if current_w == W:
        sys.stdout.write("Yes")
        return
    
    # Slide the window from index 1 to n-L
    for i in range(1, n - L + 1):
        # Update the count: remove the character leaving the window and add new character coming in.
        if s[i - 1] == 'w':
            current_w -= 1
        if s[i + L - 1] == 'w':
            current_w += 1
        if current_w == W:
            sys.stdout.write("Yes")
            return
    sys.stdout.write("No")
    
if __name__ == '__main__':
    main()