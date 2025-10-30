# YOUR CODE HERE
a, b, c = map(int, input().split())
if b <= a < c or a < c <= b < 24 or b < 24 <= a < c:
    print('Yes')
else:
    print('No')