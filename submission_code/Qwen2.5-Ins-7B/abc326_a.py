# YOUR CODE HERE
x, y = map(int, input().split())
if abs(x - y) <= 2 or (x in [1, 2, 3] and y in [1, 2, 3]) or (x in [98, 99, 100] and y in [98, 99, 100]):
    print('Yes')
else:
    print('No')