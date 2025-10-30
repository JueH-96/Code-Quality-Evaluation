def ring_dist(a, b, N):
    diff = abs(a - b)
    return min(diff, N - diff)

def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    nq_line = data[0].split()
    N = int(nq_line[0])
    Q = int(nq_line[1])
    instructions = []
    for i in range(1, 1 + Q):
        parts = data[i].split()
        H = parts[0]
        T = int(parts[1])
        instructions.append((H, T))
    
    INF = 10**15
    dp0 = [INF] * (N + 1)
    dp1 = [INF] * (N + 1)
    
    H0, T0 = instructions[0]
    if H0 == 'L':
        dist = ring_dist(1, T0, N)
        if 2 == T0:
            n1 = 2 + 1
            if n1 > N:
                n1 = 1
            n2 = 2 - 1
            if n2 < 1:
                n2 = N
            if n1 != T0:
                dp0[n1] = dist + 1
            if n2 != T0:
                dp0[n2] = dist + 1
        else:
            dp0[2] = dist
    else:
        dist = ring_dist(2, T0, N)
        if 1 == T0:
            n1 = 1 + 1
            if n1 > N:
                n1 = 1
            n2 = 1 - 1
            if n2 < 1:
                n2 = N
            if n1 != T0:
                dp1[n1] = dist + 1
            if n2 != T0:
                dp1[n2] = dist + 1
        else:
            dp1[1] = dist

    for i in range(1, Q):
        H, T_next = instructions[i]
        T_prev = instructions[i - 1][1]
        next0 = [INF] * (N + 1)
        next1 = [INF] * (N + 1)
        
        if H == 'L':
            for j in range(1, N + 1):
                if dp0[j] < INF:
                    base_cost = ring_dist(T_prev, T_next, N)
                    if j == T_next:
                        base_cost += 1
                    if j != T_next:
                        total_val = dp0[j] + base_cost
                        if total_val < next0[j]:
                            next0[j] = total_val
                    else:
                        n1 = j + 1
                        if n1 > N:
                            n1 = 1
                        n2 = j - 1
                        if n2 < 1:
                            n2 = N
                        total_val = dp0[j] + base_cost
                        if n1 != T_prev:
                            if total_val < next0[n1]:
                                next0[n1] = total_val
                        if n2 != T_prev:
                            if total_val < next0[n2]:
                                next0[n2] = total_val
            for j in range(1, N + 1):
                if dp1[j] < INF:
                    base_cost = ring_dist(j, T_next, N)
                    if T_prev == T_next:
                        base_cost += 1
                    if T_prev != T_next:
                        total_val = dp1[j] + base_cost
                        if total_val < next0[T_prev]:
                            next0[T_prev] = total_val
                    else:
                        n1 = T_prev + 1
                        if n1 > N:
                            n1 = 1
                        n2 = T_prev - 1
                        if n2 < 1:
                            n2 = N
                        total_val = dp1[j] + base_cost
                        if n1 != j:
                            if total_val < next0[n1]:
                                next0[n1] = total_val
                        if n2 != j:
                            if total_val < next0[n2]:
                                next0[n2] = total_val
        else:
            for j in range(1, N + 1):
                if dp0[j] < INF:
                    base_cost = ring_dist(j, T_next, N)
                    if T_prev == T_next:
                        base_cost += 1
                    if T_prev != T_next:
                        total_val = dp0[j] + base_cost
                        if total_val < next1[T_prev]:
                            next1[T_prev] = total_val
                    else:
                        n1 = T_prev + 1
                        if n1 > N:
                            n1 = 1
                        n2 = T_prev - 1
                        if n2 < 1:
                            n2 = N
                        total_val = dp0[j] + base_cost
                        if n1 != j:
                            if total_val < next1[n1]:
                                next1[n1] = total_val
                        if n2 != j:
                            if total_val < next1[n2]:
                                next1[n2] = total_val
            for j in range(1, N + 1):
                if dp1[j] < INF:
                    base_cost = ring_dist(T_prev, T_next, N)
                    if j == T_next:
                        base_cost += 1
                    if j != T_next:
                        total_val = dp1[j] + base_cost
                        if total_val < next1[j]:
                            next1[j] = total_val
                    else:
                        n1 = j + 1
                        if n1 > N:
                            n1 = 1
                        n2 = j - 1
                        if n2 < 1:
                            n2 = N
                        total_val = dp1[j] + base_cost
                        if n1 != T_prev:
                            if total_val < next1[n1]:
                                next1[n1] = total_val
                        if n2 != T_prev:
                            if total_val < next1[n2]:
                                next1[n2] = total_val
        dp0 = next0
        dp1 = next1
    
    ans = min(min(dp0[1:]), min(dp1[1:]))
    print(ans)

if __name__ == "__main__":
    main()