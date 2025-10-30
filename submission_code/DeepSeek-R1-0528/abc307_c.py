def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        print("No")
        return
    
    idx = 0
    H_A, W_A = map(int, data[idx].split())
    idx += 1
    A = data[idx:idx+H_A]
    idx += H_A
    
    H_B, W_B = map(int, data[idx].split())
    idx += 1
    B = data[idx:idx+H_B]
    idx += H_B
    
    H_X, W_X = map(int, data[idx].split())
    idx += 1
    X_list = data[idx:idx+H_X]
    
    blackA = []
    for i in range(H_A):
        for j in range(W_A):
            if A[i][j] == '#':
                blackA.append((i, j))
                
    blackB = []
    for i in range(H_B):
        for j in range(W_B):
            if B[i][j] == '#':
                blackB.append((i, j))
                
    for dx in range(-30, 31):
        for dy in range(-30, 31):
            combined_set = set(blackA)
            for (i, j) in blackB:
                combined_set.add((i + dx, j + dy))
                
            if not combined_set:
                continue
                
            all_i = [p[0] for p in combined_set]
            all_j = [p[1] for p in combined_set]
            min_i = min(all_i)
            max_i = max(all_i)
            min_j = min(all_j)
            max_j = max(all_j)
            
            if (max_i - min_i + 1) > H_X or (max_j - min_j + 1) > W_X:
                continue
                
            for i0 in range(max_i - H_X + 1, min_i + 1):
                for j0 in range(max_j - W_X + 1, min_j + 1):
                    pattern = []
                    for r in range(H_X):
                        s = ''
                        for c in range(W_X):
                            x = i0 + r
                            y = j0 + c
                            if (x, y) in combined_set:
                                s += '#'
                            else:
                                s += '.'
                        pattern.append(s)
                    if pattern == X_list:
                        print("Yes")
                        return
                        
    print("No")

if __name__ == '__main__':
    main()