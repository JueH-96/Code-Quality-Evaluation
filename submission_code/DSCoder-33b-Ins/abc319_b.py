# YOUR CODE HERE
N = int(input())
s = ""
for i in range(N+1):
    found = False
    for j in range(1, 10):
        if i % j == 0 and N % j == 0:
            s += str(j)
            found = True
            break
    if not found:
        s += "-"
print(s)