# YOUR CODE HERE
import sys
import sys
def solve():
    import sys
    import sys
    MOD = 998244353
    N,*rest = map(int, sys.stdin.read().split())
    balls = []
    for i in range(N):
        X, Y = rest[2*i], rest[2*i+1]
        balls.append( (X, Y) )
    # Sort balls by X ascending
    balls_sorted = sorted(balls, key=lambda x: x[0])
    P = [0] + [ball[1] for ball in balls_sorted]  # 1-indexed
    dp_prev = [0]*(N+2)  # m from 1 to N+1
    dp_prev[N+1] =1
    for k in range(1, N+1):
        dp_curr = [0]*(N+2)
        for m_prev in range(1, N+2):
            if dp_prev[m_prev] ==0:
                continue
            if m_prev >= P[k]:
                # Option1: include k
                if m_prev == N+1:
                    m_new = P[k]
                else:
                    m_new = min(m_prev, P[k])
                dp_curr[m_new] = (dp_curr[m_new] + dp_prev[m_prev]) % MOD
                # Option2: not include k
                dp_curr[m_prev] = (dp_curr[m_prev] + dp_prev[m_prev]) % MOD
        dp_prev = dp_curr
    total = sum(dp_prev[1:N+2]) % MOD
    # Subtract the empty set
    total = (total -1) % MOD
    print(total)