def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    C = list(map(int, data[2:2+N]))
    
    # Precompute cost to make S alternating starting with 0 or 1
    prefix_cost0 = [0] * (N + 1)
    prefix_cost1 = [0] * (N + 1)
    for j in range(1, N + 1):
        if j % 2 == 1:
            Tj_for_0 = '0'
            Tj_for_1 = '1'
        else:
            Tj_for_0 = '1'
            Tj_for_1 = '0'
        prefix_cost0[j] = prefix_cost0[j-1] + (C[j-1] if S[j-1] != Tj_for_0 else 0)
        prefix_cost1[j] = prefix_cost1[j-1] + (C[j-1] if S[j-1] != Tj_for_1 else 0)
    
    # Precompute suffix cost to make S alternating starting with 0 or 1
    suffix_cost0 = [0] * (N + 2)
    suffix_cost1 = [0] * (N + 2)
    for j in range(N, 0, -1):
        if j % 2 == 1:
            Tj_for_0 = '0'
            Tj_for_1 = '1'
        else:
            Tj_for_0 = '1'
            Tj_for_1 = '0'
        suffix_cost0[j] = suffix_cost0[j+1] + (C[j-1] if S[j-1] != Tj_for_0 else 0)
        suffix_cost1[j] = suffix_cost1[j+1] + (C[j-1] if S[j-1] != Tj_for_1 else 0)
    
    min_cost = float('inf')
    for i in range(1, N):
        for T1 in '01':
            if T1 == '0':
                T_i = '0' if i % 2 == 1 else '1'
            else:
                T_i = '1' if i % 2 == 1 else '0'
            T_i1 = T_i
            if T_i1 == '0':
                suffix_cost = suffix_cost0[i+1]
            else:
                suffix_cost = suffix_cost1[i+1]
            if T1 == '0':
                prefix_cost = prefix_cost0[i]
            else:
                prefix_cost = prefix_cost1[i]
            total_cost = prefix_cost + suffix_cost
            min_cost = min(min_cost, total_cost)
    
    print(min_cost)

if __name__ == '__main__':
    main()