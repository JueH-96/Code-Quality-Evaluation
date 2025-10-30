# YOUR CODE HERE
A = []
while True:
    try:
        A.append(int(input()))
    except EOFError:
        break
A.reverse()
for a in A:
    print(a)