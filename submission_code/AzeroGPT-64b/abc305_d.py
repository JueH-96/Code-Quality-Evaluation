N = int(input())
A = list(map(int, input().split()))
Q = int(input())
lr = [list(map(int, input().split())) for _ in range(Q)]

nl = (N - 1) // 2
sleep = [0] * (N + 1)
for i in range(nl):
    sleep[A[2 * i + 1]] += A[2 * i + 2] - A[2 * i + 1]
    sleep[A[2 * i + 2]] -= A[2 * i + 2] - A[2 * i + 1]

for i in range(1, N + 1):
    sleep[i] += sleep[i - 1]

ans = [sleep[r] - sleep[l] for l, r in lr]

print('
'.join(map(str, ans)))