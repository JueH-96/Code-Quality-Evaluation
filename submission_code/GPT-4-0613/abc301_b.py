N = int(input().strip())
A = list(map(int, input().strip().split()))

B = [A[0]]
for i in range(1, N):
    if abs(A[i] - A[i-1]) > 1:
        B += list(range(min(A[i], A[i-1])+1, max(A[i], A[i-1]))) + [A[i]]
    else:
        B.append(A[i])

print(' '.join(map(str, B)))