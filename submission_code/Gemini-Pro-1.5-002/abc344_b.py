# YOUR CODE HERE
a = []
while True:
    try:
        n = int(input())
        a.append(n)
    except EOFError:
        break

for i in range(len(a) - 1, -1, -1):
    print(a[i])