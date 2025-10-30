def solve():
    t = int(input().strip())
    for _ in range(t):
        S = input().strip()
        X = input().strip()
        Y = input().strip()
        if X.count('0') == Y.count('0') and X.count('1') == Y.count('1'):
            print('Yes')
        else:
            print('No')

solve()