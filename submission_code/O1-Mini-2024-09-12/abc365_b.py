N, *A = map(int, open(0).read().split())
max_val = max(A)
second_max = max(x for x in A if x != max_val)
print(A.index(second_max) + 1)