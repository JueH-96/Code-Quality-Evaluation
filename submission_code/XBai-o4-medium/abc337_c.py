n = int(input())
A = list(map(int, input().split()))
next_p = [0] * (n + 1)
front = -1

for i in range(1, n + 1):
    a_i = A[i - 1]
    if a_i == -1:
        front = i
    else:
        next_p[a_i] = i

res = []
current = front
while current != 0:
    res.append(current)
    current = next_p[current]

print(' '.join(map(str, res)))