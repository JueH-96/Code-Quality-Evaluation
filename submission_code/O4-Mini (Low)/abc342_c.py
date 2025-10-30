def main():
    import sys
    input = sys.stdin.readline
    
    # Read input
    N = int(input().strip())
    S = input().strip()
    Q = int(input().strip())
    
    # f[x] = current mapped character index for original letter x (0..25)
    f = list(range(26))
    # groups[c] = list of original-letter indices x whose current mapping f[x] == c
    groups = [[i] for i in range(26)]
    
    for _ in range(Q):
        c, d = input().split()
        ci = ord(c) - 97
        di = ord(d) - 97
        if ci == di:
            continue
        # Move all original letters mapped to ci into group di
        grp = groups[ci]
        if not grp:
            continue
        for x in grp:
            f[x] = di
            groups[di].append(x)
        # Clear the old group
        groups[ci].clear()
    
    # Build the final string
    res_chars = []
    for ch in S:
        idx = ord(ch) - 97
        mapped = f[idx]
        res_chars.append(chr(mapped + 97))
    
    sys.stdout.write("".join(res_chars))

if __name__ == "__main__":
    main()