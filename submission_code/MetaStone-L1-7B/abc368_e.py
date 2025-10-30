import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    X1 = int(data[2])
    
    trains = []
    index = 3
    for _ in range(M):
        A = int(data[index])
        B = int(data[index+1])
        S = int(data[index+2])
        T = int(data[index+3])
        trains.append((A, B, S, T))
        index += 4
    
    # For each u, collect i's where B_i == u
    from collections import defaultdict
    u_to_is = defaultdict(list)
    for i in range(M):
        u_to_is[trains[i][1]].append(i)
    
    # For each u, sort the list of i's by T_i and compute prefix_max
    prefix_max = dict()
    for u in u_to_is:
        i_list = u_to_is[u]
        i_list.sort(key=lambda x: trains[x][3])  # Sort by T_i
        prefix = []
        current_max = 0
        for i in i_list:
            val = trains[i][2] + X1 if i == 0 else (trains[i][2] + X[i])
            current_max = max(current_max, val)
            prefix.append(current_max)
        prefix_max[u] = prefix
    
    X = [0] * M
    X[0] = X1
    
    for u in u_to_is:
        i_list = u_to_is[u]
        i_list.sort(key=lambda x: trains[x][3])  # Ensure sorted by T_i
        
        # Reset X for i's
        for i in i_list:
            X[i] = 0
        
        # Process j's where A_j == u
        j_list = []
        for j in range(M):
            if trains[j][0] == u:
                j_list.append(j)
        
        for j in j_list:
            S_j = trains[j][2]
            # Find the largest index in i_list where T_i <= S_j
            T_list = [trains[i][3] for i in i_list]
            idx = bisect.bisect_right(T_list, S_j) - 1
            if idx >= 0:
                max_val = prefix_max[u][idx]
            else:
                max_val = 0
            X_j = max(max_val - S_j, 0)
            X[j] = X_j
    
    print(' '.join(map(str, X[1:M+1])))

if __name__ == '__main__':
    main()