from bisect import bisect_left

def add(sparse, i):
    x = sparse[i] = sparse[i+1] = i + 1
    s = i * 2 + 1
    while s < len(sparse):
        sparse[s] = max(sparse[s], x)
        s += s & -s

def query(sparse, i, j):
    x = 0
    while i < j:
        l = i & -i
        r = j & -j
        x = max(x, sparse[i], sparse[j])
        i += l
        j -= r
    return x

def f(lims):
    holes = []
    holeidx = {}
    sparse = [0] * (len(lims) * 2 + 1)
    for hole in lims:
        i = bisect_left(lims, hole)
        holeidx[hole] = i
        add(sparse, i)
    return holeidx, sparse

def solve(H, W, N, lims):
    holeidx, sparse = f(lims)
    ans = 0
    for px in range(1, H+1):
        for py in range(1, W+1):
            def check(k):
                x, y = px + k, py + k
                if x > H or y > W:
                    return False
                for dx in range(k):
                    if (x := holeidx.get((px + dx, py + k), W + px + dx)) < y:
                        return False
                    if (y := holeidx.get((px + k, py + dx), H + py + dx)) < x:
                        return False
                return True
            k = bisect_left(range(min(H-px, W-py)+1), False, key=check)
            if k:
                sidelen = k + query(sparse, holeidx.get((px, py), W + px), holeidx.get((px + k, py + k), H + py + k)) + 1
                ans += sidelen ** 2 - k ** 2
    return ans

solve(*open(0).read().split(), [])