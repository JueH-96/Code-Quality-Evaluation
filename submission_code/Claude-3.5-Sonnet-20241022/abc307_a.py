N = int(input())
A = list(map(int, input().split()))

result = []
for i in range(N):
    week_sum = sum(A[i*7:(i+1)*7])
    result.append(str(week_sum))

print(' '.join(result))