N = int(input())
A = list(map(int, input().split()))

count = 0
for i in range(N-2):
    for j in range(i+1, N-1):
        if A[i] == A[j]:
            continue
        for k in range(j+1, N):
            if A[i] == A[k]:
                count += 1

print(count)