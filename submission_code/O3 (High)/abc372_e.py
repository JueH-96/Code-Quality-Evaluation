import sys

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)

    N = int(next(it))
    Q = int(next(it))

    parent = list(range(N + 1))          # parent[x] = parent of x in the DSU tree
    comp_size = [1] * (N + 1)            # size of each component (valid for roots)
    top = [[i] for i in range(N + 1)]    # for each root keep the 10 largest vertices (descending)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # union by size: make ra the larger root
        if comp_size[ra] < comp_size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        comp_size[ra] += comp_size[rb]

        # merge the two top-lists and keep only 10 largest values
        merged = sorted(top[ra] + top[rb], reverse=True)[:10]
        top[ra] = merged
        # top[rb] no longer needed but we can leave it as is

    out_lines = []

    for _ in range(Q):
        typ = int(next(it))
        if typ == 1:
            u = int(next(it))
            v = int(next(it))
            union(u, v)
        else:  # typ == 2
            v = int(next(it))
            k = int(next(it))
            r = find(v)
            lst = top[r]
            if k <= len(lst):
                out_lines.append(str(lst[k - 1]))
            else:
                out_lines.append("-1")

    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()