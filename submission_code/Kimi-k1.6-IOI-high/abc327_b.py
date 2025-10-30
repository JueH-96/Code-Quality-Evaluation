B = int(input())
A = 1
while True:
    current = A ** A
    if current == B:
        print(A)
        break
    elif current > B:
        print(-1)
        break
    else:
        A += 1