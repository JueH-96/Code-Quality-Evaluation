B = int(input())
for A in range(1, 100):
    current = A ** A
    if current == B:
        print(A)
        exit()
    elif current > B:
        break
print(-1)