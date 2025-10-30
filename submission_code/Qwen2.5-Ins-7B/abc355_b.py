# YOUR CODE HERE
n, m = map(int, input().split())
a = sorted(map(int, input().split()))
b = sorted(map(int, input().split()))
c = sorted(a + b)
for i in range(len(c) - 1):
    if c[i] in a and c[i+1] in a:
        print('Yes')
        break
else:
    print('No')