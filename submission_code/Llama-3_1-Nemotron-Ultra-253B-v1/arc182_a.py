MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    P = []
    V = []
    for _ in range(Q):
        p = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        P.append(p)
        V.append(v)
    
    from collections import defaultdict

    dp = defaultdict(int)
    initial_left = [(1, 0)]
    initial_right = [(1, 0)]
    dp[(tuple(initial_left), tuple(initial_right))] = 1

    for i in range(Q):
        p = P[i]
        v = V[i]
        new_dp = defaultdict(int)
        for (left, right), cnt in dp.items():
            left_max_1 = left[0][1]
            right_max_p = 0
            for (pos, val) in right:
                if pos <= p:
                    right_max_p = max(right_max_p, val)
                else:
                    break
            max_prefix = max(left_max_1, right_max_p)
            if max_prefix <= v:
                new_left = []
                low, high, best = 1, N, 0
                while low <= high:
                    mid = (low + high) // 2
                    val = 0
                    for (l_pos, l_val) in left:
                        if l_pos <= mid:
                            val = max(val, l_val)
                        else:
                            break
                    if val >= v:
                        best = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                new_left = []
                current_pos, current_val = 1, 0
                for (l_pos, l_val) in left:
                    if l_pos > best:
                        break
                    if current_val != l_val:
                        new_left.append((current_pos, l_val))
                        current_val = l_val
                    current_pos = l_pos + 1
                if best < p:
                    new_left.append((best + 1, v))
                    current_pos = p + 1
                for (l_pos, l_val) in left:
                    if l_pos >= current_pos:
                        new_left.append((l_pos, l_val))
                merged = []
                for pos, val in new_left:
                    if merged and merged[-1][1] == val:
                        merged[-1] = (pos, val)
                    else:
                        merged.append((pos, val))
                new_left_tuple = tuple(merged)
                new_dp[(new_left_tuple, right)] = (new_dp[(new_left_tuple, right)] + cnt) % MOD
            
            right_max_N = right[-1][1]
            left_max_p = 0
            for (pos, val) in left:
                if pos <= p:
                    left_max_p = max(left_max_p, val)
                else:
                    break
            max_suffix = max(left_max_p, right_max_N)
            if max_suffix <= v:
                new_right = []
                low, high, best = p, N, N + 1
                while low <= high:
                    mid = (low + high) // 2
                    val = 0
                    for (r_pos, r_val) in right:
                        if r_pos <= mid:
                            val = max(val, r_val)
                        else:
                            break
                    if val >= v:
                        best = mid
                        high = mid - 1
                    else:
                        low = mid + 1
                new_right = []
                current_pos, current_val = 1, 0
                for (r_pos, r_val) in right:
                    if r_pos < p:
                        if current_val != r_val:
                            new_right.append((current_pos, r_val))
                            current_val = r_val
                        current_pos = r_pos + 1
                    else:
                        break
                if p <= best - 1:
                    new_right.append((p, v))
                    current_pos = best
                for (r_pos, r_val) in right:
                    if r_pos >= current_pos:
                        new_right.append((r_pos, r_val))
                merged = []
                for pos, val in new_right:
                    if merged and merged[-1][1] == val:
                        merged[-1] = (pos, val)
                    else:
                        merged.append((pos, val))
                new_right_tuple = tuple(merged)
                new_dp[(left, new_right_tuple)] = (new_dp[(left, new_right_tuple)] + cnt) % MOD
        dp = new_dp
    
    print(sum(dp.values()) % MOD)

if __name__ == '__main__':
    main()