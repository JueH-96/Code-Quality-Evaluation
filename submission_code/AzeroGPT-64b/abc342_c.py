n = int(input())
s = list(input())
q = int(input())
ops = [input().split() for _ in range(q)]

def update(i, x, y):
    next_idx = inter[x][i + 1] if i + 1 < len(inter[x]) else 2 * n + 1
    
    while i < next_idx:
        i0, i1, m = tree[i]
        char = s[i0]
        if char == y:
            return
        if char == x:
            s[i0] = y
            m[y] += 1
            m[x] -= 1
            
        if next_idx - i0 == 1:
            i = i0 + 1
        else:
            tree[i] = (i0, i1, {k:v for k, v in m.items() if v != 0})
            if i & 1:
                update(i - 1, x, y)
                break
            else:
                update(2 * i + 1, x, y)
                update(2 * i + 2, x, y)
                break

def build(s):
    m = collections.defaultdict(int)
    for c in s:
        m[c] += 1
    
    return (-1, 2 * n + 1, m)

n <<= 1
tree, inter = [None] * (4 * n), collections.defaultdict(list)

def segment_build(s):
    def build(st, ed, idx):
        if ed - st == 1:
            tree[idx] = build(st)
            return
        tree[idx] = build(st, (st + ed) // 2, 2 * idx + 1) + build((st + ed) // 2, ed, 2 * idx + 2)
    build(0, n, 0)

def query(interval):
    st, ed, interval = 0, n, sorted(interval)
    def query(st, ed, idx):
        i0, i1, m = tree[idx]
        for c, v in m.items():
            if v != 0:
                inter[c].append(i0)
        if i1 - i0 == 1 or (st <= i0 <= ed <= i1 and len(interval) == 1):
            return tree[idx]
        if len(interval) == 2:
            if interval[1] - interval[0] == 1:
                return tree[2 * idx + 1], tree[2 * idx + 2]
            ret1 = query(st, (st + ed) // 2, 2 * idx + 1)
            ret2 = query((st + ed) // 2, ed, 2 * idx + 2)
            for c, v in ret1[2].items():
                if v != 0:
                    inter[c].append(ret1[0])
            for c, v in ret2[2].items():
                if v != 0:
                    inter[c].append(ret2[0])
            return ret1[0], ret2[1], {}
        else:
            return query(st, (st + ed) // 2, 2 * idx + 1), query((st + ed) // 2, ed, 2 * idx + 2)

    query(st, ed, 0)
    zipped = zip(*sorted(inter.items()))
    inter.clear()
    return list(zipped)[1]

segment_build(s)
queries = [query((i * n, (i + 1) * n)) for i in range(2)]
start = 0

for c, d in ops:
    start = bisect_right(queries[0], start)
    update(start, c, d)
        
print(*s, sep="")