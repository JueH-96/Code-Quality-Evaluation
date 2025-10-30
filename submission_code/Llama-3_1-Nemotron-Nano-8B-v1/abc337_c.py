n = int(input())
A = list(map(int, input().split()))

front_person = None
for i in range(n):
    if A[i] == -1:
        front_person = i + 1
        break

next_person = [None] * (n + 1)
for i in range(n):
    a_i = A[i]
    if a_i != -1:
        predecessor = a_i
        current_person = i + 1
        next_person[predecessor] = current_person

current = front_person
result = []
while current is not None:
    result.append(current)
    current = next_person[current]

print(' '.join(map(str, result)))