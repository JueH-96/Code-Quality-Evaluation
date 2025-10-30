N, L, R = map(int, input().split())
A = list(range(1, N + 1))
sub_array = A[L-1:R]
sub_array.reverse()
A[L-1:R] = sub_array
print(*A)