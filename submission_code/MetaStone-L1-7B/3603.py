import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    n = int(data[ptr])
    ptr += 1
    parent = list(map(int, data[ptr:ptr + n]))
    ptr += n
    s = list(data[ptr])
    ptr += 1

    from collections import defaultdict

    children = defaultdict(list)
    for i in range(1, n):
        p = parent[i]
        children[p].append(i)

    is_pal = [False] * n
    first_char = [None] * n
    last_char = [None] * n

    def dfs(u):
        if not children[u]:
            is_pal[u] = True
            first_char[u] = s[u]
            last_char[u] = s[u]
            return s[u]
        res = []
        for v in children[u]:
            res.append(dfs(v))
        if not res:
            return s[u]
        concat = ''.join(res)
        if not concat:
            concat = s[u]
        else:
            concat += s[u]
        if len(concat) == 0:
            concat = s[u]
        else:
            concat += s[u]
        first = concat[0] if concat else s[u]
        last = concat[-1] if concat else s[u]
        if first != last:
            return None
        for c in concat:
            if c != concat[-c::-1][0]:
                return None
        if first != s[u]:
            return None
        if all_pal = True
        for v in children[u]:
            if not is_pal[v]:
                return None
        if concat != concat[::-1]:
            return None
        is_pal[u] = True
        first_char[u] = first
        last_char[u] = last
        return concat

    for u in range(n):
        dfs(u)

    res = []
    for u in range(n):
        if is_pal[u]:
            res.append("Yes")
        else:
            res.append("No")
    print(' '.join(res))

if __name__ == '__main__':
    main()