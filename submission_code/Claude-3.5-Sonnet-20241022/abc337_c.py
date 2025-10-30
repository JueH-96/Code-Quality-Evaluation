N = int(input())
A = list(map(int, input().split()))

# Find the person at front (-1)
front = A.index(-1) + 1

# Create array to store result
result = [0] * N
pos = 0
result[pos] = front

# Follow the chain from front to back
curr = front
for i in range(N-1):
    # Find who is behind current person
    for j in range(N):
        if A[j] == curr:
            pos += 1
            result[pos] = j + 1
            curr = j + 1
            break

print(*result)