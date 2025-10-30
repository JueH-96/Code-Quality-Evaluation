N = int(input())
A = list(map(int, input().split()))
Q = int(input())

sleep = [0]
for i in range(1, N, 2):
    sleep.append(sleep[-1] + A[i] - A[i-1])
for i in range(2, N, 2):
    sleep.append(sleep[-1] + A[i] - A[i-1])

for _ in range(Q):
    l, r = map(int, input().split())
    ans = 0
    for i in range(1, len(sleep)):
        if sleep[i-1] >= l and sleep[i] <= r:
            ans += sleep[i] - sleep[i-1]
        elif sleep[i-1] >= l and sleep[i] > r:
            ans += r - sleep[i-1]
        elif sleep[i-1] < l and sleep[i] <= r:
            ans += sleep[i] - l
        elif sleep[i-1] < l and sleep[i] > r:
            ans += r - l
    print(ans)