from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    
    occ = defaultdict(list)
    for idx, num in enumerate(a, 1):  # 1-based index
        occ[num].append(idx)
    
    total = 0
    for x in occ:
        lst = occ[x]
        m = len(lst)
        if m < 2:
            continue
        
        # Compute sum_part1
        sum_part1 = 0
        prefix_sum = [0] * m
        prefix_sum[0] = lst[0]
        for i in range(1, m):
            prefix_sum[i] = prefix_sum[i-1] + lst[i]
        
        for j in range(1, m):
            current_pos = lst[j]
            sum_prev = prefix_sum[j-1]
            term = j * current_pos - sum_prev - j
            sum_part1 += term
        
        # Compute sum_part2
        if m >= 3:
            sum_part2 = m * (m-1) * (m-2) // 6
        else:
            sum_part2 = 0
        
        total += (sum_part1 - sum_part2)
    
    print(total)

if __name__ == "__main__":
    main()