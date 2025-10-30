#!/usr/bin/env python3
import sys
import threading

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    strings = data[1:]
    # We'll build a trie where each node has:
    #  - children: dict from character to node index
    #  - minlen: minimum length among inserted strings through this node
    #
    # For each new string S of length L, we walk down the trie along S
    # as far as possible.  At depth l (1-based), suppose we're at node v.
    # The best cost to transform S into some previous string that shares
    # this prefix is:
    #    (L - l) deletes + (minlen[v] - l) adds
    # We take the minimum over all l we can traverse, and also the cost
    # to go to the empty string (which is L deletes).
    #
    # After reporting the cost, we insert S into the trie, updating minlen
    # along the path.
    
    INF = 10**9
    children = [{}]      # list of dicts: node -> {char: child_node}
    minlen = [INF]      # minlen[v] is minimal length of any inserted string through v
    
    out = []
    idx = 0
    for S in strings:
        L = len(S)
        # 1) Query
        best = L  # cost to make it empty by deleting all chars
        node = 0
        # traverse as far as we can along S
        for i, ch in enumerate(S):
            if ch not in children[node]:
                break
            node = children[node][ch]
            l = i + 1
            ml = minlen[node]
            # cost = (L - l) deletes + (ml - l) adds
            cost = (L - l) + (ml - l)
            if cost < best:
                best = cost
        out.append(str(best))
        
        # 2) Insert S, updating minlen
        node = 0
        for ch in S:
            nxt = children[node].get(ch)
            if nxt is None:
                nxt = len(children)
                children[node][ch] = nxt
                children.append({})
                minlen.append(INF)
            node = nxt
            if minlen[node] > L:
                minlen[node] = L
    
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()