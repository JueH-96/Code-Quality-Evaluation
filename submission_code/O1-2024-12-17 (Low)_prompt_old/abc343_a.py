def solve():
    A, B = map(int, input().split())
    S = A + B
    for n in range(10):
        if n != S:
            print(n)
            return

# Call solve() to execute
solve()