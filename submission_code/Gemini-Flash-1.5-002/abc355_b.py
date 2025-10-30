# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

c = sorted(a + b)

found = False
for i in range(len(c) - 1):
    if c[i] in a and c[i+1] in a:
        found = True
        break

if found:
    print("Yes")
else:
    print("No")