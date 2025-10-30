B = int(input())

# Check for A from 1 to 60 (more than enough given the constraints)
for A in range(1, 61):
    if A ** A == B:
        print(A)
        exit()

print(-1)