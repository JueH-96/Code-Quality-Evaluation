# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
x = 0
for i in a:
    c = 0
    for j in range(2, int(i**0.5)+1):
        if i % j == 0:
            c += 1
            if i // j != j:
                c += 1
    if i == 2:
        c = 1
    x ^= c
if x:
    print("Anna")
else:
    print("Bruno")