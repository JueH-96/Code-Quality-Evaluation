import heapq
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    X1 = int(next(it))
    
    trains = []
    for _ in range(M):
        A = int(next(it))
        B = int(next(it))
        S = int(next(it))
        T = int(next(it))
        trains.append((A, B, S, T))
    
    INF_NEG = -10**18
    best = [INF_NEG] * (N + 1)
    
    dp = [0] * M
    dp[0] = X1
    
    events = []
    A0, B0, S0, T0 = trains[0]
    heapq.heappush(events, (T0, 0, 0, B0))
    
    for i in range(1, M):
        Ai, Bi, Si, Ti = trains[i]
        heapq.heappush(events, (Si, 1, i, Ai))
    
    while events:
        time_val, etype, idx, city = heapq.heappop(events)
        if etype == 0:
            val = dp[idx] + time_val
            if val > best[city]:
                best[city] = val
        else:
            candidate = best[city]
            if candidate == INF_NEG:
                dp[idx] = 0
            else:
                required = candidate - time_val
                if required < 0:
                    dp[idx] = 0
                else:
                    dp[idx] = required
            _, B_i, _, T_i = trains[idx]
            heapq.heappush(events, (T_i, 0, idx, B_i))
    
    res = []
    for i in range(1, M):
        res.append(str(dp[i]))
    print(" ".join(res))

if __name__ == "__main__":
    main()