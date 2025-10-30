def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data or len(data) < 3:
        return
    
    K = int(data[0])  # K is provided, and per the problem K=1.
    s = data[1].rstrip("
")
    t = data[2].rstrip("
")
    
    # If S and T are already identical, no operations are needed.
    if s == t:
        sys.stdout.write("Yes")
        return
    
    # If the difference in lengths is greater than 1, it's impossible to fix with one edit.
    if abs(len(s) - len(t)) > 1:
        sys.stdout.write("No")
        return
    
    # Case when lengths are equal: check if one substitution can fix the mismatch.
    if len(s) == len(t):
        diff_count = 0
        for c1, c2 in zip(s, t):
            if c1 != c2:
                diff_count += 1
                if diff_count > 1:  # More than one different character, so not possible.
                    sys.stdout.write("No")
                    return
        # Exactly one substitution is allowed (0 is already handled above).
        sys.stdout.write("Yes" if diff_count == 1 else "No")
        return
    
    # Case when the lengths differ by exactly 1: check if one insertion or deletion is sufficient.
    # Determine which string is shorter.
    if len(s) < len(t):
        small, large = s, t
    else:
        small, large = t, s
    
    i, j = 0, 0
    found_diff = False
    while i < len(small) and j < len(large):
        if small[i] == large[j]:
            i += 1
            j += 1
        else:
            if found_diff:
                sys.stdout.write("No")
                return
            found_diff = True
            # Skip one character from the larger string.
            j += 1
    
    # If we finished the loop, the strings differ by at most one edit.
    sys.stdout.write("Yes")

if __name__ == '__main__':
    main()