B = int(input())
A = 1
while True:
    current = A ** A
    if current == B:
        print(A)
        exit()
    if current > B:
        break
    A += 1
print(-1)