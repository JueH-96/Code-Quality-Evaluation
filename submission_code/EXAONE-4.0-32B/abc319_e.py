import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    X = int(next(it))
    Y = int(next(it))
    P_list = []
    T_list = []
    for _ in range(n-1):
        p_val = int(next(it))
        t_val = int(next(it))
        P_list.append(p_val)
        T_list.append(t_val)
    
    q = int(next(it))
    queries = [int(next(it)) for _ in range(q)]
    
    M = 840
    F = [0] * M
    
    for r in range(M):
        curr = r
        total = r
        for i in range(len(P_list)):
            p = P_list[i]
            t_val = T_list[i]
            rem = curr % p
            if rem == 0:
                w = 0
            else:
                w = p - rem
            total += w + t_val
            curr = (curr + w + t_val) % M
        F[r] = total
        
    out_lines = []
    for qi in queries:
        base_time = qi + X
        r0 = base_time % M
        a = base_time // M
        ans = a * M + F[r0] + Y
        out_lines.append(str(ans))
        
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()