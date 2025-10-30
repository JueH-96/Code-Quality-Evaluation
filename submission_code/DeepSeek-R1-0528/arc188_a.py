mod = 998244353

import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        print(0)
        return
    parts = data[0].split()
    if not parts:
        print(0)
        return
    try:
        N = int(parts[0])
        K = int(parts[1])
        S = data[1].strip()
    except:
        print(0)
        return

    DP = defaultdict(int)
    DP[(1,0,0,0,0)] = 1

    for i in range(N):
        next_DP = defaultdict(int)
        for key, count_val in DP.items():
            c0, c1, c2, c3, s = key
            ch = S[i]
            if ch == '?':
                choices = ['A','B','C']
            else:
                choices = [ch]
                
            for c in choices:
                x = s // 2
                y = s % 2
                if c == 'A':
                    nx = (x + 1) % 2
                    ny = y
                elif c == 'B':
                    nx = x
                    ny = (y + 1) % 2
                else:
                    nx = (x + 1) % 2
                    ny = (y + 1) % 2
                ns = nx * 2 + ny

                if ns == 0:
                    new_key = (c0+1, c1, c2, c3, ns)
                elif ns == 1:
                    new_key = (c0, c1+1, c2, c3, ns)
                elif ns == 2:
                    new_key = (c0, c1, c2+1, c3, ns)
                else:
                    new_key = (c0, c1, c2, c3+1, ns)
                    
                next_DP[new_key] = (next_DP[new_key] + count_val) % mod

        DP = next_DP

    total_ans = 0
    for key, count_val in DP.items():
        c0, c1, c2, c3, s = key
        total_good = c0*(c0-1)//2 + c1*(c1-1)//2 + c2*(c2-1)//2 + c3*(c3-1)//2
        if total_good >= K:
            total_ans = (total_ans + count_val) % mod

    print(total_ans % mod)

if __name__ == '__main__':
    main()