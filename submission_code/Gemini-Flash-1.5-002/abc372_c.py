# YOUR CODE HERE
n, q = map(int, input().split())
s = list(input())
for _ in range(q):
    x, c = input().split()
    s[int(x) - 1] = c
    count = 0
    for i in range(n - 2):
        if s[i:i + 3] == ['A', 'B', 'C']:
            count += 1
    print(count)