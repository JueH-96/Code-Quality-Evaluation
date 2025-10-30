import itertools

def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    
    H = int(data[0])
    W = int(data[1])
    idx = 2
    A = []
    for i in range(H):
        row = list(map(int, data[idx:idx+W]))
        idx += W
        A.append(row)
        
    B = []
    for i in range(H):
        row = list(map(int, data[idx:idx+W]))
        idx += W
        B.append(row)
        
    flatA = [num for row in A for num in row]
    flatB = [num for row in B for num in row]
    if sorted(flatA) != sorted(flatB):
        print(-1)
        return
        
    row_perms = list(itertools.permutations(range(H)))
    col_perms = list(itertools.permutations(range(W)))
    
    row_inv_dict = {}
    for p in row_perms:
        arr = list(p)
        cnt = 0
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                if arr[i] > arr[j]:
                    cnt += 1
        row_inv_dict[p] = cnt
        
    col_inv_dict = {}
    for p in col_perms:
        arr = list(p)
        cnt = 0
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                if arr[i] > arr[j]:
                    cnt += 1
        col_inv_dict[p] = cnt
        
    min_ops = 10**9
    found = False
    for rp in row_perms:
        for cp in col_perms:
            valid = True
            for i in range(H):
                for j in range(W):
                    if A[rp[i]][cp[j]] != B[i][j]:
                        valid = False
                        break
                if not valid:
                    break
                    
            if valid:
                found = True
                total_ops = row_inv_dict[rp] + col_inv_dict[cp]
                if total_ops < min_ops:
                    min_ops = total_ops
                    
    if found:
        print(min_ops)
    else:
        print(-1)

if __name__ == "__main__":
    main()