def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    value_indices = {}
    for index in range(n):
        value = a[index]
        if value not in value_indices:
            value_indices[value] = []
        value_indices[value].append(index + 1)
        
    total_triples = 0
    for value in value_indices:
        indices_list = value_indices[value]
        m = len(indices_list)
        if m < 2:
            continue
        for r_index in range(m):
            for s_index in range(r_index + 1, m):
                i = indices_list[r_index]
                k = indices_list[s_index]
                valid_j_count = (k - i - 1) - (s_index - r_index - 1)
                if valid_j_count < 0:
                    valid_j_count = 0
                total_triples += valid_j_count
                
    print(total_triples)

if __name__ == '__main__':
    solve()