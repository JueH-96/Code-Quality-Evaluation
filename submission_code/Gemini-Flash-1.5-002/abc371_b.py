# YOUR CODE HERE
n, m = map(int, input().split())
eldest_son = [None] * (n + 1)
for i in range(m):
    a, b = input().split()
    a = int(a)
    if b == 'M':
        if eldest_son[a] is None:
            print('Yes')
            eldest_son[a] = True
        else:
            print('No')
    else:
        print('No')