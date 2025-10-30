import random
from typing import List

# Assuming standard recursion depth or judge increases it.
# import sys
# sys.setrecursionlimit(2000) # Might need adjustment depending on environment and test cases

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        
        # Build adjacency list for children, already sorted by child index
        children = [[] for _ in range(n)]
        # Children are added in increasing order of their node numbers
        # because we iterate i from 1 to n-1.
        for i in range(1, n):
            children[parent[i]].append(i)

        # Using two large prime moduli and two bases for hashing
        # Using fixed small prime bases, common in competitive programming
        m1, m2 = 1_000_000_007, 1_000_000_009
        p1, p2 = 31, 37 
        # Note: For stronger hashing against malicious tests, use random primes as bases
        # p1 = random.randint(1000, 10000) # Example random range
        # p2 = random.randint(1000, 10000)
        # while p2 == p1:
        #     p2 = random.randint(1000, 10000)

        # Precompute powers of p1 and p2 modulo m1 and m2
        p1_pow = [1] * (n + 1)
        p2_pow = [1] * (n + 1)
        for i in range(1, n + 1):
            p1_pow[i] = (p1_pow[i-1] * p1) % m1
            p2_pow[i] = (p2_pow[i-1] * p2) % m2

        answer = [False] * n

        # DFS function to compute subtree size, hash, and reverse hash
        # Returns (subtree_len, h1, h2, rh1, rh2)
        def dfs_hash(u):
            # Get character value (1-indexed 'a'->1, 'b'->2, ...)
            char_val = ord(s[u]) - ord('a') + 1

            # Base case: leaf node
            if not children[u]:
                # String is just the single character s[u]. Always a palindrome.
                answer[u] = True
                # Length, h1, h2, rh1, rh2
                return (1, char_val, char_val, char_val, char_val) 

            # Recursive step for non-leaf node
            subtree_len = 1
            # Hashes for the concatenated strings from children S(c_1)...S(c_k)
            h_children_part_1, h_children_part_2 = 0, 0 
            
            children_results = []
            for v in children[u]:
                children_results.append(dfs_hash(v))

            # Combine children results in increasing order of child index for forward hash
            # S(u) = S(c_1) S(c_2) ... S(c_k) s[u]
            for v_res in children_results:
                v_len, v_h1, v_h2, v_rh1, v_rh2 = v_res
                # h(Prefix + S(v)) = (h(Prefix) * p^len(S(v)) + h(S(v))) mod m
                h_children_part_1 = (h_children_part_1 * p1_pow[v_len] + v_h1) % m1
                h_children_part_2 = (h_children_part_2 * p2_pow[v_len] + v_h2) % m2
                subtree_len += v_len

            # Final hash for S(u) = (h(S(c_1)...S(c_k)) * p^1 + s[u]) mod m
            final_h1 = (h_children_part_1 * p1_pow[1] + char_val) % m1
            final_h2 = (h_children_part_2 * p2_pow[1] + char_val) % m2

            # Combine children results in decreasing order of child index for reverse hash
            # reverse(S(u)) = s[u] reverse(S(c_k)) ... reverse(S(c_1))
            # Start with hash of s[u]
            current_rh1 = char_val
            current_rh2 = char_val
            
            # Iterate children results in reverse order (corresponds to children in decreasing index order c_k, ..., c_1)
            for v_res in reversed(children_results):
                 v_len, v_h1, v_h2, v_rh1, v_rh2 = v_res # v_rh1, v_rh2 are hashes for S(v)[::-1]
                 # rh(Current_Reversed_Prefix + reverse(S(v))) = (rh(Current_Reversed_Prefix) * p^len(S(v)) + rh(S(v)[::-1])) mod m
                 current_rh1 = (current_rh1 * p1_pow[v_len] + v_rh1) % m1
                 current_rh2 = (current_rh2 * p2_pow[v_len] + v_rh2) % m2

            final_rh1, final_rh2 = current_rh1, current_rh2

            # Check if S(u) is a palindrome by comparing its hash with the hash of its reverse
            if final_h1 == final_rh1 and final_h2 == final_rh2:
                answer[u] = True
            else:
                answer[u] = False

            # Return results for the parent node
            return (subtree_len, final_h1, final_h2, final_rh1, final_rh2)

        # Start DFS from root (node 0)
        # The results for node 0 are computed and stored in `answer[0]` by the call.
        # We don't strictly need the return value from the top-level call.
        dfs_hash(0)

        return answer