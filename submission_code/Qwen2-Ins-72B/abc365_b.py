N = int(input())
A = list(map(int, input().split()))

max_index = A.index(max(A))
A[max_index] = 0
second_max_index = A.index(max(A))

print(second_max_index + 1)