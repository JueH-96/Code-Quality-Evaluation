import sys
import bisect

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it)); w = int(next(it))
    blocks = []
    for i in range(n):
        x = int(next(it)); y = int(next(it))
        blocks.append((x, y))
    
    groups = [[] for _ in range(w+1)]
    for (x, y) in blocks:
        if 1 <= x <= w:
            groups[x].append(y)
    
    for i in range(1, w+1):
        groups[i].sort()
    
    min_size = 10**9
    for i in range(1, w+1):
        if len(groups[i]) < min_size:
            min_size = len(groups[i])
    
    if min_size == 0:
        max_val = []
    else:
        max_val = [-10**18] * min_size
        for x in range(1, w+1):
            if len(groups[x]) < min_size:
                continue
            for r in range(min_size):
                value = groups[x][r] - r
                if value > max_val[r]:
                    max_val[r] = value

    q = int(next(it))
    out_lines = []
    for i in range(q):
        t = int(next(it)); a = int(next(it))
        a_idx = a - 1
        if a_idx < 0 or a_idx >= n:
            out_lines.append("Yes")
        else:
            x, y = blocks[a_idx]
            if x < 1 or x > w:
                out_lines.append("Yes")
            else:
                list_x = groups[x]
                r = bisect.bisect_left(list_x, y)
                if r >= min_size:
                    out_lines.append("Yes")
                else:
                    L = y - r
                    if t < L:
                        out_lines.append("Yes")
                    else:
                        if max_val[r] > t:
                            out_lines.append("Yes")
                        else:
                            out_lines.append("No")
    
    for line in out_lines:
        print(line)

if __name__ == '__main__':
    main()