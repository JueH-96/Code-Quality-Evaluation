N = int(input())
people = []
min_age = float('inf')
min_index = -1
for i in range(N):
    S, A = input().split()
    A = int(A)
    people.append((S, A))
    if A < min_age:
        min_age = A
        min_index = i

for i in range(N):
    print(people[(min_index + i) % N][0])