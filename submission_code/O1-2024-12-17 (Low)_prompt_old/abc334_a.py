def solve():
    import sys
    B, G = map(int, sys.stdin.readline().split())
    if B > G:
        print("Bat")
    else:
        print("Glove")

solve()