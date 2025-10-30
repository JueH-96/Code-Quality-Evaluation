N = int(input())
A = list(map(int, input().split()))

indexes = [0]*(N+1)
middle_indexes = [0]*(N+1)
for i, a in enumerate(A):
    if indexes[a] == 1:
        middle_indexes[a] = i
    indexes[a] += 1

sorted_indexes = sorted(range(1, N+1), key=lambda x: middle_indexes[x])
print(' '.join(map(str, sorted_indexes)))