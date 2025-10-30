def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(m)]
    
    # Find the earliest occurrence (L) for each element in B from A.
    L = [-1] * m
    pos = 0
    for i in range(m):
        # Find B[i] in A starting from pos.
        while pos < n and A[pos] != B[i]:
            pos += 1
        if pos == n:
            # B is not a subsequence of A at all.
            sys.stdout.write("No")
            return
        L[i] = pos
        pos += 1  # move past the found element
    
    # Find the latest occurrence (R) for each element in B from A.
    R = [-1] * m
    pos = n - 1
    for i in range(m - 1, -1, -1):
        while pos >= 0 and A[pos] != B[i]:
            pos -= 1
        if pos < 0:
            sys.stdout.write("No")
            return
        R[i] = pos
        pos -= 1  # move to the left
    
    # Check if for any B[i], we have an alternative choice in A.
    # That is, if L[i] < R[i] for any i, then we have two distinct choices.
    for i in range(m):
        if L[i] < R[i]:
            sys.stdout.write("Yes")
            return
    
    sys.stdout.write("No")

if __name__ == '__main__':
    main()