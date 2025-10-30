import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    regular = []
    pull = []
    openers = []
    
    for _ in range(N):
        t = int(input[ptr])
        ptr += 1
        x = int(input[ptr])
        ptr += 1
        if t == 0:
            pull.append(x)
        elif t == 1:
            regular.append(x)
        else:
            openers.append(x)
    
    # Sort and compute prefix sums
    regular.sort(reverse=True)
    pull.sort(reverse=True)
    openers.sort(reverse=True)
    
    # prefix sums for regular
    reg_sum = [0]
    for x in regular:
        reg_sum.append(reg_sum[-1] + x)
    
    # prefix sums for pull
    pull_sum = [0]
    for x in pull:
        pull_sum.append(pull_sum[-1] + x)
    
    # prefix sums for opener capacities
    opener_cap = [0]
    for x in openers:
        opener_cap.append(opener_cap[-1] + x)
    
    max_happiness = 0
    
    max_o = min(M, len(openers))
    for o in range(0, max_o + 1):
        available = M - o
        if available < 0:
            continue
        current_cap = opener_cap[o]
        s_upper = min(current_cap, len(regular), available)
        low = 0
        high = s_upper
        best = 0
        
        # Ternary search
        for _ in range(100):
            if high - low < 3:
                break
            m1 = low + (high - low) // 3
            m2 = high - (high - low) // 3
            # compute f1
            s_pull1 = available - m1
            s_pull1 = min(s_pull1, len(pull))
            f1 = reg_sum[m1] + pull_sum[s_pull1]
            # compute f2
            s_pull2 = available - m2
            s_pull2 = min(s_pull2, len(pull))
            f2 = reg_sum[m2] + pull_sum[s_pull2]
            if f1 < f2:
                low = m1
            else:
                high = m2
        
        # Check all in [low, high]
        for s in range(low, high + 1):
            if s > s_upper:
                continue
            s_pull = available - s
            s_pull = min(s_pull, len(pull))
            current_val = reg_sum[s] + pull_sum[s_pull]
            if current_val > best:
                best = current_val
        
        if best > max_happiness:
            max_happiness = best
    
    print(max_happiness)
    
if __name__ == '__main__':
    main()