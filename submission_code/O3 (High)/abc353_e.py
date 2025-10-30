import sys

def main() -> None:
    # Read every token (numbers and strings) at once
    tokens = sys.stdin.buffer.read().split()
    if not tokens:
        return
    n = int(tokens[0])
    # the next n tokens are the strings themselves
    strs = [tok.decode() for tok in tokens[1:1 + n]]

    # Trie represented with two parallel lists:
    #   children[i] – a dict mapping character to next-node index
    #   cnt[i]      – how many strings pass through node i
    children = []
    cnt = []

    def new_node() -> int:
        """create a new trie node and return its index"""
        children.append({})
        cnt.append(0)
        return len(cnt) - 1

    root = new_node()          # index 0

    # Build the trie and fill cnt[]
    for s in strs:
        node = root
        cnt[node] += 1         # root is passed by every string
        for ch in s:
            nxt = children[node].get(ch)
            if nxt is None:
                nxt = new_node()
                children[node][ch] = nxt
            node = nxt
            cnt[node] += 1

    # The answer is  Σ  C(cnt[v], 2)  over every non-root node v
    # Every pair that shares a prefix represented by v contributes 1
    # to the LCP length, hence we sum over all prefix nodes.
    ans = 0
    for c in cnt[1:]:          # skip root (depth 0 contributes nothing)
        ans += c * (c - 1) // 2

    print(ans)

if __name__ == "__main__":
    main()