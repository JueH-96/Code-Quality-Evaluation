import itertools

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    h = int(data[idx])
    w = int(data[idx+1])
    idx +=2
    
    a = []
    for _ in range(h):
        row = list(map(int, data[idx:idx+w]))
        a.append(row)
        idx +=w
    
    b = []
    for _ in range(h):
        row = list(map(int, data[idx:idx+w]))
        b.append(row)
        idx +=w
    
    flat_a = [num for row in a for num in row]
    flat_b = [num for row in b for num in row]
    
    if sorted(flat_a) != sorted(flat_b):
        print(-1)
        return
    
    min_ops = float('inf')
    
    def count_inversion(perm):
        cnt = 0
        for i in range(len(perm)):
            for j in range(i+1, len(perm)):
                if perm[i] > perm[j]:
                    cnt +=1
        return cnt
    
    for p in itertools.permutations(range(h)):
        for q in itertools.permutations(range(w)):
            valid = True
            for i in range(h):
                for j in range(w):
                    if a[p[i]][q[j]] != b[i][j]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                row_swaps = count_inversion(p)
                col_swaps = count_inversion(q)
                total = row_swaps + col_swaps
                if total < min_ops:
                    min_ops = total
    
    if min_ops != float('inf'):
        print(min_ops)
    else:
        print(-1)

if __name__ == "__main__":
    main()