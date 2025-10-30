N = int(input())
people = []
for i in range(N):
    S, A = input().split()
    A = int(A)
    people.append((A, i, S))
people.sort()
for _, i, S in people:
    print(S)