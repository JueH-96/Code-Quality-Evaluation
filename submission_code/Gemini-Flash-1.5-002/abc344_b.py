# YOUR CODE HERE
l = []
while True:
    try:
        x = int(input())
        l.append(x)
    except EOFError:
        break
for i in range(len(l) - 1, -1, -1):
    print(l[i])