import sys

B = int(sys.stdin.readline())

for A in range(1, 61):
    current = A ** A
    if current == B:
        print(A)
        sys.exit()
    elif current > B:
        break

print(-1)