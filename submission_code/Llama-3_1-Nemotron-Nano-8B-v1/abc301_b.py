n = int(input())
A = list(map(int, input().split()))

while True:
    found = False
    for i in range(len(A) - 1):
        a, b = A[i], A[i+1]
        if abs(a - b) != 1:
            if a < b:
                insert = list(range(a+1, b))
            else:
                insert = list(range(a-1, b, -1))
            A = A[:i+1] + insert + A[i+1:]
            found = True
            break
    if not found:
        break

print(' '.join(map(str, A)))