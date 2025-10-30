def solve():
    import sys
    data = sys.stdin.read().strip().split()
    A, B = map(int, data)
    print((A ** B) + (B ** A))

solve()