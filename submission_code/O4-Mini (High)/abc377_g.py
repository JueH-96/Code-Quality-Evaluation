def main():
    import sys
    input = sys.stdin.readline
    INF = 10**9
    N = int(input())
    # children: list of dicts, each dict maps char -> node index
    children = [{}]
    # min_len[v]: minimal length of any string that goes through node v
    min_len = [INF]
    out = []
    for _ in range(N):
        S = input().rstrip('
')
        L = len(S)
        # Query phase: walk as far as possible in the trie, track best f = 2*d - min_len[node]
        node = 0
        f_max = 0
        d = 0
        for ch in S:
            d += 1
            nxt = children[node].get(ch)
            if nxt is None:
                break
            node = nxt
            # compute candidate f at this depth
            tmp = (d << 1) - min_len[node]
            if tmp > f_max:
                f_max = tmp
        # ensure f_max >= 0 (empty string target gives f=0)
        if f_max < 0:
            f_max = 0
        ans = L - f_max
        out.append(str(ans))
        # Insert phase: add S to trie, updating min_len along the path
        node = 0
        if L < min_len[node]:
            min_len[node] = L
        for ch in S:
            nxt = children[node].get(ch)
            if nxt is None:
                nxt = len(children)
                children[node][ch] = nxt
                children.append({})
                min_len.append(INF)
            node = nxt
            if L < min_len[node]:
                min_len[node] = L
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()