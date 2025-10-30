from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    S = input[ptr]
    ptr += 1
    
    C = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    color_counts = defaultdict(int)
    for color in C:
        color_counts[color] += 1
    
    sorted_colors = sorted(color_counts.keys())
    prefix_sum = [0] * (len(sorted_colors) + 1)
    for i in range(len(sorted_colors)-1, -1, -1):
        prefix_sum[i] = prefix_sum[i+1] + color_counts[sorted_colors[i]]
    
    cnt = {}
    for i in range(len(sorted_colors)):
        color = sorted_colors[i]
        cnt[color] = prefix_sum[i]
    
    groups = defaultdict(list)
    for j in range(N):
        color = C[j]
        groups[color].append(j)
    
    res = [''] * N
    for color in range(1, M+1):
        if color not in color_counts:
            continue
        group = groups[color]
        group_elements = [S[j] for j in group]
        k = cnt[color]
        rotated_group = group_elements[-k:] + group_elements[:-k]
        for idx, pos in enumerate(group):
            res[pos] = rotated_group[idx]
    
    print(''.join(res))

if __name__ == '__main__':
    main()