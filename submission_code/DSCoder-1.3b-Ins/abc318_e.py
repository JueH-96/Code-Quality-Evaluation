# YOUR CODE HERE

N = int(input().strip())
A = list(map(int, input().strip().split()))

count = 0

for i in range(N):
    for j in range(i+1, N):
        if A[i] == A[j]:
            for k in range(j+1, N):
                if A[i] != A[k]:
                    count += 1

print(count)