def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    P = [int(next(it)) for _ in range(n)]
    # Build a mapping from person number to their position in line
    pos = [0] * (n + 1)
    for idx, person in enumerate(P, start=1):
        pos[person] = idx
    q = int(next(it))
    out = []
    for _ in range(q):
        a = int(next(it))
        b = int(next(it))
        # Compare positions and append the one closer to the front
        if pos[a] < pos[b]:
            out.append(str(a))
        else:
            out.append(str(b))
    sys.stdout.write("
".join(out))

# Call main to execute        
main()