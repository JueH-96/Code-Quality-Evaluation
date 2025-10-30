N = int(input())
A = list(map(int, input().split()))

result = []
for i in range(N):
    # Calculate sum of all elements greater than A[i]
    sum_greater = sum(x for x in A if x > A[i])
    result.append(str(sum_greater))

print(' '.join(result))