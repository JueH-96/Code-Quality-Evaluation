import sys


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin.buffer.read().splitlines()
    n = int(data[0])
    strings = [line.decode().strip() for line in data[1:n + 1]]

    INF = 10 ** 18

    # Trie implemented with adjacency lists
    first = [-1]          # first edge index for each node
    best_len = [INF]      # minimal length of any string in the subtree of the node

    to = []               # destination node of each edge
    ch = []               # character (0-25) of each edge
    nxt = []              # next edge index in the linked list of a node

    def find_child(node: int, c: int) -> int:
        """return child node via character c or -1"""
        e = first[node]
        while e != -1:
            if ch[e] == c:
                return to[e]
            e = nxt[e]
        return -1

    def add_child(node: int, c: int) -> int:
        """create child via character c and return its id"""
        child = len(first)
        first.append(-1)
        best_len.append(INF)

        edge_idx = len(to)
        to.append(child)
        ch.append(c)
        nxt.append(first[node])
        first[node] = edge_idx
        return child

    out = []

    for s in strings:
        L = len(s)
        ans = L                      # delete all -> empty string
        node = 0
        depth = 0

        # consider root (common prefix length 0)
        if best_len[node] != INF:
            ans = min(ans, L + best_len[node])

        # walk along the current string while the path already exists
        for character in s:
            c_idx = ord(character) - 97
            child = find_child(node, c_idx)
            if child == -1:
                break                # prefix ends here
            node = child
            depth += 1
            if best_len[node] != INF:
                ans = min(ans, L + best_len[node] - 2 * depth)

        out.append(str(ans))

        # insert the current string into the trie (update best_len on the way)
        node = 0
        if L < best_len[node]:
            best_len[node] = L
        for character in s:
            c_idx = ord(character) - 97
            child = find_child(node, c_idx)
            if child == -1:
                child = add_child(node, c_idx)
            node = child
            if L < best_len[node]:
                best_len[node] = L

    sys.stdout.write('
'.join(out))


if __name__ == "__main__":
    main()