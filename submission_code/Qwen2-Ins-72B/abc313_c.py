N = int(input())
A = list(map(int, input().split()))

total = sum(A)
avg = total // N
remainder = total % N

A = [a - avg for a in A]
A.sort()

left = sum([abs(a) for a in A[:remainder]])
right = sum([abs(a) for a in A[remainder:]])

print(min(left, right))