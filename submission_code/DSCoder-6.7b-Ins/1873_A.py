# YOUR CODE HERE

def solve():
    t = int(input().strip())
    for _ in range(t):
        s = input().strip()
        if s == 'abc' or s == 'acb' or s == 'bac' or s == 'cab' or s == 'cba':
            print('YES')
        else:
            print('NO')

solve()