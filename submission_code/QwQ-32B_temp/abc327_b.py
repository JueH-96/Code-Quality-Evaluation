import sys

B = int(input())

for A in range(1, 61):
    product = 1
    for _ in range(A):
        product *= A
        if product > B:
            break
    if product == B:
        print(A)
        sys.exit()
    elif product > B:
        print(-1)
        sys.exit()

print(-1)