B = int(input())
A = 1
result = -1
while True:
    current = A ** A
    if current == B:
        result = A
        break
    elif current > B:
        break
    A += 1
print(result)