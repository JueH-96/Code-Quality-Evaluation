def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx += 2
    
    pull_tabs = []
    regular_cans = []
    can_openers = []
    
    for _ in range(N):
        T = int(data[idx])
        X = int(data[idx+1])
        idx += 2
        if T == 0:
            pull_tabs.append(X)
        elif T == 1:
            regular_cans.append(X)
        else:
            can_openers.append(X)
    
    # Sort each list in descending order
    pull_tabs.sort(reverse=True)
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)
    
    # Compute prefix sums
    pull_prefix = [0]
    for x in pull_tabs:
        pull_prefix.append(pull_prefix[-1] + x)
    
    reg_prefix = [0]
    for x in regular_cans:
        reg_prefix.append(reg_prefix[-1] + x)
    
    can_open_prefix = [0]
    for x in can_openers:
        can_open_prefix.append(can_open_prefix[-1] + x)
    
    max_happiness = 0
    max_k = min(len(can_openers), M)
    
    for k in range(0, max_k + 1):
        if k > M:
            continue
        can_open_sum = can_open_prefix[k]
        rem = M - k
        if rem < 0:
            continue
        
        # Part A: s <= can_open_sum
        s_max_a = min(can_open_sum, rem, len(regular_cans))
        best_a = 0
        if s_max_a >= 0:
            upper = min(s_max_a, 200)
            for s in range(0, upper + 1):
                t = rem - s
                if t < 0:
                    continue
                t = min(t, len(pull_tabs))
                sum_pull = pull_prefix[t] if t <= len(pull_tabs) else (pull_prefix[-1] if pull_prefix else 0)
                s_reg = min(s, can_open_sum)
                s_reg = min(s_reg, len(regular_cans))
                sum_reg = reg_prefix[s_reg] if s_reg <= len(regular_cans) else (reg_prefix[-1] if reg_prefix else 0)
                current = sum_pull + sum_reg
                if current > best_a:
                    best_a = current
            if s_max_a > upper:
                s = s_max_a
                t = rem - s
                if t >= 0:
                    t = min(t, len(pull_tabs))
                    sum_pull = pull_prefix[t] if t <= len(pull_tabs) else (pull_prefix[-1] if pull_prefix else 0)
                    s_reg = min(s, can_open_sum)
                    s_reg = min(s_reg, len(regular_cans))
                    sum_reg = reg_prefix[s_reg] if s_reg <= len(regular_cans) else (reg_prefix[-1] if reg_prefix else 0)
                    current = sum_pull + sum_reg
                    if current > best_a:
                        best_a = current
        
        # Part B: s > can_open_sum
        max_b = 0
        s_start = can_open_sum + 1
        s_end = min(rem, len(regular_cans))
        if s_start <= s_end:
            s = s_start
            t = rem - s
            if t < 0:
                sum_pull = 0
            else:
                t = min(t, len(pull_tabs))
                sum_pull = pull_prefix[t] if t <= len(pull_tabs) else (pull_prefix[-1] if pull_prefix else 0)
            sum_reg = reg_prefix[can_open_sum] if can_open_sum <= len(regular_cans) else (reg_prefix[-1] if reg_prefix else 0)
            current_b = sum_pull + sum_reg
            if current_b > max_b:
                max_b = current_b
        
        current_max = max(best_a, max_b)
        if current_max > max_happiness:
            max_happiness = current_max
    
    print(max_happiness)

if __name__ == '__main__':
    main()