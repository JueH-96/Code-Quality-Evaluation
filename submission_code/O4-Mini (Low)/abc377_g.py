def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    
    N = int(input())
    # Trie nodes: each node has a dict of children and a min_len value
    # We store children as a dict: char -> node index
    children = []
    min_len = []
    # create root
    children.append({})
    min_len.append(10**9)
    node_count = 1  # next free node index

    for _ in range(N):
        s = input().rstrip('
')
        L = len(s)
        # Query phase: traverse trie along s, track best X = 2*i - min_len_S
        maxX = 0
        node = 0
        # depth i from 1 to L
        for i, ch in enumerate(s, start=1):
            if ch not in children[node]:
                break
            node = children[node][ch]
            # at this node, all inserted strings sharing this prefix have length >= min_len[node]
            mn = min_len[node]
            # candidate X
            x = 2*i - mn
            if x > maxX:
                maxX = x
        # minimal cost = min(len(s), len(s) - maxX)
        cost = L - maxX
        if cost > L:
            cost = L
        # print answer
        print(cost)
        
        # Insert phase: add s into trie, update min_len along path
        node = 0
        if min_len[0] > L:
            min_len[0] = L
        for ch in s:
            nxt = children[node].get(ch)
            if nxt is None:
                nxt = node_count
                node_count += 1
                children[node][ch] = nxt
                children.append({})
                min_len.append(10**9)
            # update min_len at nxt
            if min_len[nxt] > L:
                min_len[nxt] = L
            node = nxt

if __name__ == "__main__":
    main()