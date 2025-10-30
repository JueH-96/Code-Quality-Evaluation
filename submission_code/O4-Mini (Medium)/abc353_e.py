import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    strings = data[1:1+n]
    # Build a trie. Each node has a dict of children and a count of how many strings pass through it.
    children = []
    cnt = []
    # Initialize root node
    children.append({})
    cnt.append(0)
    # Insert each string
    for s in strings:
        u = 0
        for ch in s:
            d = children[u]
            if ch in d:
                v = d[ch]
            else:
                v = len(children)
                d[ch] = v
                children.append({})
                cnt.append(0)
            u = v
            cnt[u] += 1
    # Sum over all non-root nodes: each node at depth>=1 contributes C(cnt,2)
    ans = 0
    for c in cnt[1:]:
        if c > 1:
            ans += c*(c-1)//2
    print(ans)

if __name__ == "__main__":
    main()