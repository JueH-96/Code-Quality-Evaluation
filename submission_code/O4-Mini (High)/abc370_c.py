def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    S, T = data[0], data[1]
    n = len(S)
    # Work with a mutable list of characters
    cur = list(S)
    result = []
    # Repeat until cur equals T
    while True:
        if "".join(cur) == T:
            break
        # Find all mismatch positions
        mismatches = [i for i in range(n) if cur[i] != T[i]]
        # Try to find the leftmost i where cur[i] > T[i]
        idx = None
        for i in mismatches:
            if cur[i] > T[i]:
                idx = i
                break
        # If none found, pick the rightmost mismatch
        if idx is None:
            idx = mismatches[-1]
        # Make the change
        cur[idx] = T[idx]
        result.append("".join(cur))
    # Output
    out = []
    out.append(str(len(result)))
    out.extend(result)
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()