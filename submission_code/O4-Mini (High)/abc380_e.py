import sys
from array import array

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    # DSU arrays: parent, size, left bound, right bound, color, and color counts
    parent = array('I', range(n+1))
    size   = array('I', (1 for _ in range(n+1)))
    leftB  = array('I', range(n+1))
    rightB = array('I', range(n+1))
    color  = array('I', range(n+1))
    cnt    = array('I', (1 for _ in range(n+1)))
    cnt[0] = 0

    def find(x):
        # path‐halving
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return ra
        # union by size: attach smaller under larger
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        # merge interval bounds
        if leftB[rb] < leftB[ra]:
            leftB[ra] = leftB[rb]
        if rightB[rb] > rightB[ra]:
            rightB[ra] = rightB[rb]
        # color[ra] stays the same (both are same color)
        return ra

    out = []
    for _ in range(q):
        t = next(it)
        if t == b'1':
            x = int(next(it))
            c_new = int(next(it))
            r = find(x)
            old_c = color[r]
            if old_c != c_new:
                # update color counts
                seg = size[r]
                cnt[old_c] -= seg
                cnt[c_new] += seg
                # repaint this component
                color[r] = c_new
                # try to merge with left neighbor
                lpos = leftB[r] - 1
                if leftB[r] > 1:
                    rl = find(lpos)
                    if rl != r and color[rl] == c_new:
                        r = union(r, rl)
                # try to merge with right neighbor
                rpos = rightB[r] + 1
                if rightB[r] < n:
                    rr = find(rpos)
                    if rr != r and color[rr] == c_new:
                        r = union(r, rr)
        else:  # t == b'2'
            c = int(next(it))
            out.append(str(cnt[c]))

    sys.stdout.write("
".join(out))


if __name__ == "__main__":
    main()