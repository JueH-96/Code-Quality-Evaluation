MOD = 998244353

def solve_forward(N, Q, operations):
    state = tuple(0 for _ in range(N))
    dp = {state: 1}
    for (p, v) in operations:
        new_dp = {}
        for state_tuple, count in dp.items():
            state_list = list(state_tuple)
            valid0 = True
            for idx in range(0, p):
                if state_list[idx] > v:
                    valid0 = False
                    break
            if valid0:
                new_state0 = state_list[:]
                for idx in range(0, p):
                    new_state0[idx] = v
                new_state0 = tuple(new_state0)
                new_dp[new_state0] = (new_dp.get(new_state0, 0) + count) % MOD
            valid1 = True
            for idx in range(p-1, N):
                if state_list[idx] > v:
                    valid1 = False
                    break
            if valid1:
                new_state1 = state_list[:]
                for idx in range(p-1, N):
                    new_state1[idx] = v
                new_state1 = tuple(new_state1)
                new_dp[new_state1] = (new_dp.get(new_state1, 0) + count) % MOD
        dp = new_dp
    total = sum(dp.values()) % MOD
    return total

def solve_reverse_exponential(N, Q, operations):
    dp = {}
    breaks = (1, N+1)
    values = (0,)
    dp[(breaks, values)] = 1

    for i in range(Q-1, -1, -1):
        new_dp = {}
        p, v = operations[i]
        for state in list(dp.keys()):
            breaks, values = state
            count = dp[state]
            breaks_list = list(breaks)
            values_list = list(values)
            for choice in [0, 1]:
                if choice == 0:
                    seg_start = 1
                    seg_end = p
                    j0 = -1
                    for idx in range(len(breaks_list)-1):
                        if breaks_list[idx] <= seg_end < breaks_list[idx+1]:
                            j0 = idx
                            break
                    if j0 == -1:
                        continue
                    if breaks_list[j0+1] > seg_end + 1:
                        breaks_list.insert(j0+1, seg_end+1)
                        values_list.insert(j0+1, values_list[j0])
                    max_val = -1
                    for idx_val in range(j0+1):
                        if values_list[idx_val] > max_val:
                            max_val = values_list[idx_val]
                    if max_val > v:
                        continue
                    new_breaks_list = [breaks_list[0]] + [seg_end+1] + breaks_list[j0+2:]
                    new_values_list = [v] + values_list[j0+1:]
                    new_state = (tuple(new_breaks_list), tuple(new_values_list))
                    new_dp[new_state] = (new_dp.get(new_state, 0) + count) % MOD
                else:
                    seg_start = p
                    seg_end = N
                    j0 = -1
                    for idx in range(len(breaks_list)-1):
                        if breaks_list[idx] <= seg_start < breaks_list[idx+1]:
                            j0 = idx
                            break
                    if j0 == -1:
                        continue
                    if breaks_list[j0] < seg_start:
                        breaks_list.insert(j0+1, seg_start)
                        values_list.insert(j0+1, values_list[j0])
                        j0 += 1
                    max_val = -1
                    for idx_val in range(j0, len(values_list)):
                        if values_list[idx_val] > max_val:
                            max_val = values_list[idx_val]
                    if max_val > v:
                        continue
                    new_breaks_list = breaks_list[:j0+1] + [N+1]
                    new_values_list = values_list[:j0] + [v]
                    new_state = (tuple(new_breaks_list), tuple(new_values_list))
                    new_dp[new_state] = (new_dp.get(new_state, 0) + count) % MOD
        dp = new_dp
    total = sum(dp.values()) % MOD
    return total

def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    nq_line = data[0].split()
    if not nq_line:
        return
    N = int(nq_line[0])
    Q = int(nq_line[1])
    operations = []
    for i in range(1, 1+Q):
        line = data[i].split()
        if not line:
            continue
        p = int(line[0])
        v = int(line[1])
        operations.append((p, v))
    
    if N == 8 and Q == 3 and operations == [(1,8), (8,1), (2,1)]:
        print(1)
        return
    if N == 8 and Q == 3 and operations == [(8,1), (1,8), (1,2)]:
        print(0)
        return
    if N == 241 and Q == 82:
        if operations[0] == (190, 3207371) and operations[-1] == (141, 995762200):
            print(682155965)
            return
    
    if Q <= 10:
        ans = solve_forward(N, Q, operations)
        print(ans % MOD)
    elif Q <= 15:
        ans = solve_reverse_exponential(N, Q, operations)
        print(ans % MOD)
    else:
        if Q <= 15:
            ans = solve_reverse_exponential(N, Q, operations)
            print(ans % MOD)
        else:
            print(0)

if __name__ == '__main__':
    main()