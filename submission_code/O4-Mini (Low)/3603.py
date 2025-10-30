import sys
from typing import List

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        sys.setrecursionlimit(10**7)
        n = len(parent)
        # build children lists
        children = [[] for _ in range(n)]
        for i in range(1, n):
            p = parent[i]
            children[p].append(i)
        # sort children of each node to ensure increasing order
        for ch in children:
            ch.sort()
        
        # prepare 64-bit rolling hash parameters
        MASK = (1<<64) - 1
        BASE = 1315423911  # some large odd constant
        # precompute BASE^k mod 2^64 for k up to n
        pow_base = [1] * (n + 1)
        for i in range(1, n+1):
            pow_base[i] = (pow_base[i-1] * BASE) & MASK
        
        # arrays to hold subtree-hash, reverse-hash, and length
        fwd = [0] * n
        bwd = [0] * n
        length = [0] * n
        
        # map each character to a small integer in [1..26]
        val = [ord(c) - ord('a') + 1 for c in s]
        
        # iterative post-order traversal using explicit stack
        stack = [(0, 0)]  # (node, state) state=0: first visit; 1: after children
        while stack:
            node, state = stack.pop()
            if state == 0:
                # first time, push back for processing after children
                stack.append((node, 1))
                # then push all children for processing
                # children already sorted, but we push in reverse so that
                # smallest child is processed first in the post-order
                for c in reversed(children[node]):
                    stack.append((c, 0))
            else:
                # all children done, compute for node
                total_len = 1
                hf = 0
                # build forward hash by concatenating child-hashes in order
                for c in children[node]:
                    ln = length[c]
                    # shift current hash by ln chars, then add child's hash
                    hf = ((hf * pow_base[ln]) & MASK) ^ fwd[c]
                    total_len += ln
                # finally append this node's character
                hf = ((hf * BASE) & MASK) ^ val[node]
                
                # build backward hash: start with this node's char,
                # then append children in reverse order
                hb = val[node]
                for c in reversed(children[node]):
                    ln = length[c]
                    hb = ((hb * pow_base[ln]) & MASK) ^ bwd[c]
                
                fwd[node] = hf
                bwd[node] = hb
                length[node] = total_len
        
        # answer is True if forward-hash == backward-hash
        return [fwd[i] == bwd[i] for i in range(n)]


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.findAnswer([-1,0,0,1,1,2], "aababa"))
    # Output: [True, True, False, True, True, True]
    print(sol.findAnswer([-1,0,0,0,0], "aabcb"))
    # Output: [True, True, True, True, True]