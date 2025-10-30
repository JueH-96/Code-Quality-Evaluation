def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    for _ in range(n):
        neighbors = []
        for j in range(n):
            if next(it) == '1':
                neighbors.append(str(j+1))
        # Print neighbors in ascending order (they are already in order j=0..n-1)
        if neighbors:
            sys.stdout.write(" ".join(neighbors))
        # Even if no neighbors, we need to output a blank line
        sys.stdout.write("
")

# Call main to execute
main()