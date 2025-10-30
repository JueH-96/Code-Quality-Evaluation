def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    value_indices = {}
    for i in range(n):
        value = a[i]
        if value not in value_indices:
            value_indices[value] = []
        value_indices[value].append(i + 1)
        
    total_count = 0
    for value in value_indices:
        indices_list = value_indices[value]
        m = len(indices_list)
        if m < 2:
            continue
        for p_index in range(m):
            for q_index in range(p_index + 1, m):
                i = indices_list[p_index]
                k = indices_list[q_index]
                valid_j_count = 0
                for j in range(i + 1, k):
                    if a[j-1] != value:
                        valid_j_count += 1
                total_count += valid_j_count
                
    print(total_count)

if __name__ == '__main__':
    solve()