def solve():
    import sys
    data = sys.stdin.read().split()
    A, B = map(int, data)
    
    suspects = {1, 2, 3}
    if A in suspects:
        suspects.remove(A)
    if B in suspects:
        suspects.remove(B)
    
    if len(suspects) == 1:
        print(suspects.pop())
    else:
        print(-1)

# Call the solve() function
solve()