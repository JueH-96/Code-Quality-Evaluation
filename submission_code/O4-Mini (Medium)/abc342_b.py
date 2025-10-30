def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    
    # Read number of people
    n = int(next(it))
    # Read the lineup P (1-based indexing for convenience)
    P = [0] + [int(next(it)) for _ in range(n)]
    
    # Build an inverse mapping: pos[x] = position of person x in the line
    pos = [0] * (n + 1)
    for idx in range(1, n + 1):
        person = P[idx]
        pos[person] = idx
    
    # Read number of queries
    q = int(next(it))
    out_lines = []
    
    # Process each query
    for _ in range(q):
        a = int(next(it))
        b = int(next(it))
        # Compare positions and choose the one closer to the front (smaller index)
        if pos[a] < pos[b]:
            out_lines.append(str(a))
        else:
            out_lines.append(str(b))
    
    # Print all answers
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()