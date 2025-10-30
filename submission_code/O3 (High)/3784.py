from typing import List
from collections import defaultdict

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        
        # after removing one word we have (n-1) words left
        if n - 1 < k:                       # impossible for every removal
            return [0] * n
        
        # -----------------  build Trie  -----------------
        children = []        # list[dict] : children[node] -> {char_index : child_node}
        cnt      = []        # number of words that pass this node
        depth    = []        # depth of node (root = 0)
        uniqFrag = []        # will be filled later – flag: node is unique&fragile
        
        children.append({})
        cnt.append(0)
        depth.append(0)
        uniqFrag.append(False)          # root
        
        for w in words:
            node = 0
            for ch in w:
                c = ord(ch) - 97
                nxt = children[node].get(c)
                if nxt is None:
                    nxt = len(children)
                    children[node][c] = nxt
                    children.append({})
                    cnt.append(0)
                    depth.append(depth[node] + 1)
                    uniqFrag.append(False)
                node = nxt
                cnt[node] += 1
        
        # --------------  analyse nodes  -----------------
        depth_good_cnt = defaultdict(int)   # depth -> #nodes with cnt>=k
        depth_single   = {}                 # depth -> the (only) node id if so far unique
        
        stableMax = 0                       # deepest prefix that is always safe (cnt>=k+1)
        
        for node_id in range(1, len(cnt)):          # skip root
            c = cnt[node_id]
            if c >= k:
                d = depth[node_id]
                prev = depth_good_cnt.get(d, 0)
                depth_good_cnt[d] = prev + 1
                if prev == 0:                       # first node of this depth
                    depth_single[d] = node_id
                elif prev == 1:                     # now not unique any more
                    depth_single.pop(d, None)
                if c >= k + 1:
                    stableMax = max(stableMax, d)
        
        safeMax = stableMax                        # deepest prefix guaranteed safe
        uniqueFragDepths = []                      # depths with a single node, cnt==k
        for d, num_nodes in depth_good_cnt.items():
            if num_nodes >= 2:
                safeMax = max(safeMax, d)          # multi nodes at this depth -> safe
            else:                                  # exactly one node
                node_id = depth_single[d]
                if cnt[node_id] >= k + 1:          # unique but still cnt>k → safe
                    safeMax = max(safeMax, d)
                else:                              # unique & fragile (cnt==k)
                    uniqueFragDepths.append(d)
                    uniqFrag[node_id] = True       # mark this node
        
        if not uniqueFragDepths:                   # no fragile depths, everything fixed
            return [safeMax] * n
        
        uniqueFragDepths.sort(reverse=True)        # deepest first
        candDepths = [d for d in uniqueFragDepths if d > safeMax]
        if not candDepths:                         # every fragile depth not deeper than safe
            return [safeMax] * n
        
        # --------------  produce answers  --------------
        res = []
        for w in words:
            # collect fragile depths ( > safeMax ) that this word belongs to
            bad = set()
            node = 0
            for ch in w:
                node = children[node][ord(ch) - 97]
                if uniqFrag[node] and depth[node] > safeMax:
                    bad.add(depth[node])
            
            ans = safeMax
            for d in candDepths:               # pick first depth not forbidden
                if d not in bad:
                    ans = d
                    break
            res.append(ans)
        
        return res