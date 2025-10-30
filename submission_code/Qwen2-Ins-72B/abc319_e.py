from heapq import heappush, heappop

N, X, Y = map(int, input().split())
PT = [list(map(int, input().split())) for _ in range(N-1)]
Q = int(input())
query = [int(input()) for _ in range(Q)]

INF = 10**18
bus = [INF] * N
bus[0] = X
for i in range(N-1):
    p, t = PT[i]
    for j in range(p):
        if bus[i] + j < bus[i+1]:
            bus[i+1] = bus[i] + j
            for k in range(8):
                if bus[i+1] + k*p < bus[i+1] + t:
                    bus[i+1] += k*p
                else:
                    bus[i+1] += t
                    break

ans = []
for q in query:
    time = [q] * N
    time[0] = q + X
    que = [(q+X, 0)]
    while que:
        t, i = heappop(que)
        if time[i] < t:
            continue
        for j in range(8):
            if time[i] + j*PT[i][0] < time[i+1]:
                time[i+1] = time[i] + j*PT[i][0]
                for k in range(8):
                    if time[i+1] + k*PT[i][0] < time[i+1] + PT[i][1]:
                        time[i+1] += k*PT[i][0]
                    else:
                        time[i+1] += PT[i][1]
                        heappush(que, (time[i+1], i+1))
                        break
    ans.append(time[-1] + Y)

print(*ans, sep='
')