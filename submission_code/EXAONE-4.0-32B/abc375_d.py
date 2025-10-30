from collections import defaultdict

def main():
    S = input().strip()
    groups = defaultdict(list)
    for idx, char in enumerate(S):
        groups[char].append(idx)
    
    total_ans = 0
    for arr in groups.values():
        m = len(arr)
        if m < 2:
            continue
        T = sum(arr)
        S1 = 0
        for j in range(m):
            S1 += j * arr[j]
        total_diff = 2 * S1 - (m - 1) * T
        num_pairs = m * (m - 1) // 2
        total_ans += total_diff - num_pairs
        
    print(total_ans)

if __name__ == '__main__':
    main()