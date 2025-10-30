def solve():
    n = int(input())
    s = input()
    
    for i in range(3, n + 1):
        sub = s[:i]
        if 'A' in sub and 'B' in sub and 'C' in sub:
            print(i)
            return

solve()