n = int(input())
A = list(map(int, input().split()))
head = -1
next_person = [None] * (n + 1)  # Using a list for O(1) access

for i in range(1, n + 1):
    a_i = A[i - 1]
    if a_i == -1:
        head = i
    else:
        next_person[a_i] = i

result = []
current = head
while current is not None:
    result.append(current)
    current = next_person[current]

print(' '.join(map(str, result)))