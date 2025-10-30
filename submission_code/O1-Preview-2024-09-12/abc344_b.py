# YOUR CODE HERE
lst = []
while True:
    x = int(input())
    lst.append(x)
    if x == 0:
        break

lst.reverse()

for val in lst:
    print(val)