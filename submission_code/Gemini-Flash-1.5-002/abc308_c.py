# YOUR CODE HERE
N = int(input())
people = []
for i in range(N):
    A, B = map(int, input().split())
    if A + B == 0:
        rate = 0
    else:
        rate = A / (A + B)
    people.append((rate, i + 1))

people.sort(key=lambda x: (-x[0], x[1]))
result = [person[1] for person in people]
print(*result)