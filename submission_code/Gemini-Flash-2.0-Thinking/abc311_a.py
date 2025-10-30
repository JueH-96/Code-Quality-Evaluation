def solve():
    n = int(input())
    s = input()

    for i in range(n):
        prefix = s[:i+1]
        if 'A' in prefix and 'B' in prefix and 'C' in prefix:
            print(i + 1)
            return

solve()