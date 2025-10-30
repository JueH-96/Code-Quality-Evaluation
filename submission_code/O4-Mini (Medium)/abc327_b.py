def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    B = int(data[0])
    # We look for a positive integer A such that A^A == B.
    # Since A^A grows very fast, A will be quite small when B <= 1e18.
    # We'll just try A = 1,2,... until A^A exceeds B.
    ans = -1
    A = 1
    while True:
        # Compute A^A
        p = pow(A, A)
        if p == B:
            ans = A
            break
        if p > B:
            break
        A += 1
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()