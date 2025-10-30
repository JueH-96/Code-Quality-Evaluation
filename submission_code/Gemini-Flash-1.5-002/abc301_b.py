# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

while True:
    flag = True
    for i in range(len(a) - 1):
        if abs(a[i] - a[i+1]) != 1:
            flag = False
            break
    if flag:
        break
    
    for i in range(len(a) - 1):
        if abs(a[i] - a[i+1]) != 1:
            if a[i] < a[i+1]:
                b = list(range(a[i] + 1, a[i+1]))
                a = a[:i+1] + b + a[i+1:]
            else:
                b = list(range(a[i] - 1, a[i+1], -1))
                a = a[:i+1] + b + a[i+1:]
            break

print(*a)