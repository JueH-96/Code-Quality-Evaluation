T = int(input().strip())
for _ in range(T):
    N = int(input().strip())
    P = list(map(int, input().strip().split()))
    P = [0] + P + [0]
    cnt = 0
    for i in range(1, N+2):
        if P[i-1] > P[i]:
            cnt += 1
    print(cnt-1)