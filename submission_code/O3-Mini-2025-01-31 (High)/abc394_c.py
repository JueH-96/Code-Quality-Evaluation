def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    S = data[0].rstrip("
")
    n = len(S)
    res = []
    i = 0
    # The key observation:
    # Every contiguous block of one or more 'W' immediately followed by an 'A'
    # will eventually become "A" followed by "C" repeated (number of Ws) times.
    # Everything else stays the same.
    while i < n:
        if S[i] == 'W':
            j = i
            # Count this block of consecutive Ws.
            while j < n and S[j] == 'W':
                j += 1
            # If the block is immediately followed by an A, perform the transformation.
            if j < n and S[j] == 'A':
                count_ws = j - i  # number of Ws in the block
                res.append("A" + "C" * count_ws)
                i = j + 1  # skip over the block and the following A 
            else:
                # Otherwise, no transformation: output these Ws as they are.
                res.append(S[i:j])
                i = j
        else:
            # Characters other than 'W' are added unchanged.
            res.append(S[i])
            i += 1
    sys.stdout.write("".join(res))


if __name__ == '__main__':
    main()