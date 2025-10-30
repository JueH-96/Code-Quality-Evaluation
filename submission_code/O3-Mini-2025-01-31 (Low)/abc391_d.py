def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    W = int(next(it))
    # Prepare lists per column, storing (y, index). (Columns are 1-indexed.)
    cols = {c: [] for c in range(1, W+1)}
    # We'll store block details (x,y) for block indices 1..n.
    blocks = [None]*(n+1)
    for i in range(1, n+1):
        x = int(next(it))
        y = int(next(it))
        blocks[i] = (x, y)
        cols[x].append((y, i))
    
    # For each column, sort by y ascending.
    rank = [0]*(n+1)  # rank[block id] = rank in its column (starting at 1)
    colCount = {}
    for c in range(1, W+1):
        lst = cols[c]
        lst.sort(key=lambda tup: tup[0])
        colCount[c] = len(lst)
        for pos, (_, idx) in enumerate(lst, start=1):
            rank[idx] = pos

    # r0 is the minimum count in any column; i.e. maximum removals overall.
    r0 = None
    for c in range(1, W+1):
        cnt = colCount[c]
        if r0 is None or cnt < r0:
            r0 = cnt
    if r0 is None:
        r0 = 0  # in case there are no columns (should not happen)

    Q = int(next(it))
    out_lines = []
    # For each query:
    # Query gives T and block A.
    # At time T+0.5 the number of removals that have occurred is r = min(T-1, r0)
    # So block A exists if its rank > r.
    for _ in range(Q):
        T = int(next(it))
        A = int(next(it))
        removals = T - 1  # removals scheduled at times 1,2,...,T-1
        if removals > r0:
            removals = r0
        if rank[A] > removals:
            out_lines.append("Yes")
        else:
            out_lines.append("No")
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()