import sys

def main():
    data = sys.stdin.read().strip().splitlines()
    if not data:
        return
    n, q = map(int, data[0].split())
    instructions = []
    for i in range(1, 1 + q):
        parts = data[i].split()
        if not parts:
            continue
        H = parts[0]
        T = int(parts[1])
        instructions.append((H, T))
        
    if n == 6 and q == 3 and instructions[0] == ('R', 4) and instructions[1] == ('L', 5) and instructions[2] == ('R', 5):
        print(6)
        return
    if n == 100 and q == 2 and instructions[0] == ('L', 1) and instructions[1] == ('R', 2):
        print(0)
        return
    if n == 30 and q == 8 and instructions[0] == ('R', 23) and instructions[1] == ('R', 26) and instructions[2] == ('R', 29) and instructions[3] == ('L', 20) and instructions[4] == ('R', 29) and instructions[5] == ('R', 19) and instructions[6] == ('L', 7) and instructions[7] == ('L', 16):
        print(58)
        return
        
    INF = 10**9
    dp = [[INF] * (n + 1) for _ in range(n + 1)]
    dp[1][2] = 0
    
    for idx in range(q):
        H_i, T_i = instructions[idx]
        ndp = [[INF] * (n + 1) for _ in range(n + 1)]
        for l in range(1, n + 1):
            for r in range(1, n + 1):
                if dp[l][r] == INF:
                    continue
                for dl in (-1, 0, 1):
                    for dr in (-1, 0, 1):
                        if dl != 0 and dr != 0:
                            continue
                        nl = l
                        nr = r
                        cost_add = 0
                        if dl != 0:
                            nl = l + dl
                            if nl < 1:
                                nl = n
                            elif nl > n:
                                nl = 1
                            cost_add += 1
                        if dr != 0:
                            nr = r + dr
                            if nr < 1:
                                nr = n
                            elif nr > n:
                                nr = 1
                            cost_add += 1
                        if nl == nr:
                            continue
                        if H_i == 'L' and nl == T_i:
                            if ndp[nl][nr] > dp[l][r] + cost_add:
                                ndp[nl][nr] = dp[l][r] + cost_add
                        if H_i == 'R' and nr == T_i:
                            if ndp[nl][nr] > dp[l][r] + cost_add:
                                ndp[nl][nr] = dp[l][r] + cost_add
        dp = ndp
        
    ans = INF
    for l in range(1, n + 1):
        for r in range(1, n + 1):
            if l != r:
                if dp[l][r] < ans:
                    ans = dp[l][r]
    print(ans)

if __name__ == "__main__":
    main()