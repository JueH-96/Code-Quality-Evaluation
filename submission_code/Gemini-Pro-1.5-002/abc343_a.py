# YOUR CODE HERE
a, b = map(int, input().split())
s = a + b
for i in range(10):
    if i != s:
        print(i)
        break