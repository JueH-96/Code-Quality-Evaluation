def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    try:
        K = int(data[0])
    except:
        return
    # Read the two strings.
    if len(data) < 3:
        return
    s = data[1].rstrip("
")
    t = data[2].rstrip("
")
    n = len(s)
    m = len(t)
    # A necessary condition: the edit distance is at least |n-m|.
    if abs(n - m) > K:
        sys.stdout.write("No")
        return

    # For speed assign local variables.
    s_local = s
    t_local = t

    # d = 0: no operations. Do the “snake” along the diagonal 0.
    x = 0
    lim = n if n < m else m
    while x < lim and s_local[x] == t_local[x]:
        x += 1
    # prev will hold for a given edit cost d the farthest reached x coordinate on diagonal k,
    # where k = i - j.
    prev = {0: x}
    # If we already matched both strings then we are done.
    if x >= n and x >= m:
        sys.stdout.write("Yes")
        return

    # For d = 1 to K, try to see if we can transform s into t with d operations.
    # Allowed operations (each cost 1):
    #  • Insertion: We do not advance in s but one character in t is “used” (i remains same)
    #    so the diagonal shifts: coming from diagonal k+1.
    #  • Deletion: Advance in s but not in t, so coming from diagonal k-1 (plus one step in s).
    #  • Substitution: Advance in both s and t; coming from the same diagonal.
    #
    # Then after doing one edit, we “slide” along matching characters (the snake).
    for d in range(1, K + 1):
        curr = {}
        # For a fixed cost d the reachable diagonals k run from -d to d.
        for k in range(-d, d + 1):
            cand = -1
            # Insertion: from diagonal k+1 (i remains the same)
            if (k + 1) in prev:
                a = prev[k + 1]
                if a > cand:
                    cand = a
            # Deletion: from diagonal k-1 (advance in s by one)
            if (k - 1) in prev:
                a = prev[k - 1] + 1
                if a > cand:
                    cand = a
            # Substitution: from the same diagonal (advance in both)
            if k in prev:
                a = prev[k] + 1
                if a > cand:
                    cand = a

            x_val = cand
            y_val = x_val - k  # Because k = x - y.
            # Slide along the “snake” (match as many characters as possible).
            while x_val < n and y_val < m and s_local[x_val] == t_local[y_val]:
                x_val += 1
                y_val = x_val - k
            curr[k] = x_val
            # If we have reached the end of both s and t then we can transform s into t.
            if x_val >= n and y_val >= m:
                sys.stdout.write("Yes")
                return
        prev = curr
    sys.stdout.write("No")

if __name__ == '__main__':
    main()