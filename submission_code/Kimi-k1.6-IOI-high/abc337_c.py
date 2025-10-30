n = int(input())
A = list(map(int, input().split()))
next_arr = [0] * (n + 1)
front = -1

for i in range(1, n + 1):
    a = A[i - 1]
    if a == -1:
        front = i
    else:
        next_arr[a] = i

result = []
current = front
while current != 0:
    result.append(current)
    current = next_arr[current]

print(' '.join(map(str, result)))