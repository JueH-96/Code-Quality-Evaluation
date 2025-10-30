# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
m = int(input())
b = list(map(int, input().split()))
l = int(input())
c = list(map(int, input().split()))
q = int(input())
x = list(map(int, input().split()))

for xi in x:
    found = False
    for ai in a:
        for bi in b:
            for ci in c:
                if ai + bi + ci == xi:
                    found = True
                    break
            if found:
                break
        if found:
            break
    if found:
        print("Yes")
    else:
        print("No")