# YOUR CODE HERE

def solve():
    t = int(input())
    for _ in range(t):
        s = input()
        x = input()
        y = input()
        x_count = x.count('1')
        y_count = y.count('1')
        if x_count != y_count:
            print('No')
        else:
            print('Yes')

solve()