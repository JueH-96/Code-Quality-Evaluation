# YOUR CODE HERE
t = int(input().strip())
for _ in range(t):
    s = input().strip()
    if s == 'abc' or s == 'bac' or s == 'cba':
        print('YES')
    else:
        print('NO')