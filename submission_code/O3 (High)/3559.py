from typing import List
from collections import defaultdict

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # ---------- build a trie that contains every word ----------
        class Node:
            __slots__ = ("child",)
            def __init__(self):
                self.child = {}
        root = Node()
        for w in words:
            cur = root
            for ch in w:
                nxt = cur.child.get(ch)
                if nxt is None:
                    nxt = Node()
                    cur.child[ch] = nxt
                cur = nxt
            # no “end” marker needed – every node is a prefix by definition

        n = len(target)
        INF = n + 1                          # anything larger than possible answer
        dp = [INF] * (n + 1)                 # dp[i] = min pieces to build target[:i]
        dp[0] = 0

        # ---------- classic DP using the trie ----------
        for i in range(n):
            if dp[i] == INF:                 # position i cannot be reached – skip
                continue
            node = root
            for j in range(i, n):
                node = node.child.get(target[j])
                if node is None:             # no further prefix possible
                    break
                # substring target[i : j+1] is a prefix of some word → valid piece
                if dp[i] + 1 < dp[j + 1]:
                    dp[j + 1] = dp[i] + 1

        return -1 if dp[n] == INF else dp[n]