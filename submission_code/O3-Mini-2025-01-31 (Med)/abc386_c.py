def main():
    import sys
    data = sys.stdin.read().splitlines()
    
    # Read input data
    if not data:
        return
    K = int(data[0].strip())  # K is always 1 for our problem.
    s = data[1].strip()
    t = data[2].strip()
    
    # If the strings are identical, no operation is needed.
    if s == t:
        sys.stdout.write("Yes")
        return

    n, m = len(s), len(t)
    
    # If the lengths differ by more than 1 then at least 2 operations are needed.
    if abs(n - m) > 1:
        sys.stdout.write("No")
        return
    
    # When both strings have the same length, one substitution might work.
    if n == m:
        diff_count = 0
        for i in range(n):
            if s[i] != t[i]:
                diff_count += 1
                if diff_count > 1:
                    sys.stdout.write("No")
                    return
        # Only one substitution is needed if there is exactly one difference.
        if diff_count == 1:
            sys.stdout.write("Yes")
        else:
            sys.stdout.write("No")
        return

    # When lengths differ by 1, then one insertion or one deletion can make them equal.
    # Let's define 'longer' and 'shorter'
    if n > m:
        longer, shorter = s, t
    else:
        longer, shorter = t, s

    i = j = 0
    diff_found = False
    while i < len(longer) and j < len(shorter):
        if longer[i] == shorter[j]:
            i += 1
            j += 1
        else:
            # If we have already found one difference, then second difference means more than one operation.
            if diff_found:
                sys.stdout.write("No")
                return
            diff_found = True
            # Skip one character in the longer string, simulating an insertion/deletion.
            i += 1
    sys.stdout.write("Yes")

if __name__ == '__main__':
    main()