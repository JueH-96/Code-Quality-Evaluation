import sys
import math

def main():
    import sys
    import math

    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    coords = []
    for i in range(N):
        x = int(N_and_rest[1 + 2*i])
        y = int(N_and_rest[2 + 2*i])
        coords.append( (x,y) )
    
    MAX_C = 40
    MAX_STEP = 100
    INF = float('inf')
    dp = [ [INF]*(MAX_C+1) for _ in range(N)]
    dp[0][0] = 0.0
    
    for i in range(N):
        for c in range(MAX_C+1):
            current = dp[i][c]
            if current < INF:
                for j in range(i+1, min(N, i + MAX_STEP +1)):
                    skipped = j - i -1
                    new_c = c + skipped
                    if new_c > MAX_C:
                        continue
                    dx = coords[i][0] - coords[j][0]
                    dy = coords[i][1] - coords[j][1]
                    distance = math.hypot(dx, dy)
                    s_new = current + distance
                    if s_new < dp[j][new_c]:
                        dp[j][new_c] = s_new
    
    min_s = INF
    for c in range(MAX_C+1):
        s = dp[N-1][c]
        if s < INF:
            if c >0:
                penalty = 2**(c-1)
                total_s = s + penalty
            else:
                total_s = s
            if total_s < min_s:
                min_s = total_s
    print("{0:.12f}".format(min_s))

if __name__ == "__main__":
    main()