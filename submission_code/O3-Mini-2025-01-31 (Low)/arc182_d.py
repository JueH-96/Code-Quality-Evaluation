def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n)]
    
    operations = 0
    for a, b in zip(A, B):
        # Compute the forward distance from a to b modulo m.
        d = (b - a) % m
        # The cost is the minimum of going forward d steps or going backward (m-d) steps.
        operations += min(d, m - d)
    
    sys.stdout.write(str(operations))

if __name__ == '__main__':
    main()