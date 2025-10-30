import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    W = int(next(it))
    columns = [[] for _ in range(W + 1)]
    
    for i in range(n):
        x = int(next(it))
        y = int(next(it))
        columns[x].append((y, i))
    
    min_rank = float('inf')
    for i in range(1, W + 1):
        if len(columns[i]) < min_rank:
            min_rank = len(columns[i])
    
    for i in range(1, W + 1):
        columns[i].sort(key=lambda t: t[0])
    
    M = [0] * (min_rank + 1)
    for i in range(1, W + 1):
        lst = columns[i]
        for j in range(min_rank):
            y_val = lst[j][0]
            if y_val > M[j + 1]:
                M[j + 1] = y_val
    
    block_rank = [0] * n
    for i in range(1, W + 1):
        lst = columns[i]
        for idx_in_col, (_, idx_block) in enumerate(lst):
            block_rank[idx_block] = idx_in_col + 1
    
    q = int(next(it))
    out_lines = []
    for _ in range(q):
        t = int(next(it))
        a = int(next(it))
        block_index = a - 1
        r = block_rank[block_index]
        if r <= min_rank:
            if t < M[r]:
                out_lines.append("Yes")
            else:
                out_lines.append("No")
        else:
            out_lines.append("Yes")
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()