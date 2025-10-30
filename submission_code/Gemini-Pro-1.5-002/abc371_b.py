# YOUR CODE HERE
n, m = map(int, input().split())
taro = [False] * (n + 1)
for _ in range(m):
    a, b = input().split()
    a = int(a)
    if b == 'M' and not taro[a]:
        print('Yes')
        taro[a] = True
    else:
        print('No')