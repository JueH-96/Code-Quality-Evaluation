import bisect
from typing import List
from functools import lru_cache

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n1 = len(word1)
        n2 = len(word2)
        
        # Build occurrence lists for each letter in word1.
        occ = { chr(c): [] for c in range(ord('a'), ord('z')+1) }
        for i, ch in enumerate(word1):
            occ[ch].append(i)
            
        # Precompute next_diff: for every index in word1,
        # next_diff[i] = smallest index j > i with word1[j] different from word1[i], or n1 if none.
        next_diff = [0] * n1
        if n1:
            next_diff[n1-1] = n1
        for i in range(n1-2, -1, -1):
            if word1[i] == word1[i+1]:
                next_diff[i] = next_diff[i+1]
            else:
                next_diff[i] = i+1

        # Helper: given a starting index (pos in word1) and position j in word2,
        # returns the unique greedy sequence of indices that exactly matches word2[j:].
        # (Returns a tuple of indices or None if not possible.)
        @lru_cache(maxsize=None)
        def get_exact_seq(pos: int, j: int):
            res = []
            current = pos
            for k in range(j, n2):
                c = word2[k]
                lst = occ[c]
                idx = bisect.bisect_left(lst, current)
                if idx == len(lst):
                    return None
                chosen = lst[idx]
                res.append(chosen)
                current = chosen + 1
            return tuple(res)
        
        def feasible_exact(pos: int, j: int) -> bool:
            return get_exact_seq(pos, j) is not None
        
        # Compute the greedy prefix sequence: the lexicographically smallest exact match for word2[0:len]
        prefix_list = []
        pos_ptr = 0
        for k in range(n2):
            c = word2[k]
            lst = occ[c]
            i_pos = bisect.bisect_left(lst, pos_ptr)
            if i_pos == len(lst):
                break
            chosen = lst[i_pos]
            prefix_list.append(chosen)
            pos_ptr = chosen + 1
        L = len(prefix_list)
        
        best_seq = None
        candidate_exact = get_exact_seq(0, 0)
        if candidate_exact is not None:
            best_seq = candidate_exact

        # Try every possible “mismatch position” k (0-indexed in word2)
        # (For all positions 0 ≤ k < n2 where the prefix word2[0:k] can be exactly matched.)
        for k in range(0, n2):
            if k != 0 and k > L:
                break  # cannot even match the prefix of length k exactly.
            prefix_seq = tuple(prefix_list[:k]) if k > 0 else tuple()
            start_pos = prefix_seq[-1] + 1 if k > 0 else 0
            if start_pos >= n1:
                continue
            # We need an index i ≥ start_pos for which word1[i] != word2[k] (this uses up our 1 allowed change)
            candidate = start_pos
            if candidate < n1 and word1[candidate] == word2[k]:
                candidate = next_diff[candidate]
            while candidate < n1:
                # We have candidate with word1[candidate] != word2[k].
                if word1[candidate] != word2[k]:
                    # Check that the remaining pattern word2[k+1:] is exactly matchable from candidate+1.
                    if k == n2 - 1 or feasible_exact(candidate+1, k+1):
                        break
                candidate = next_diff[candidate] if candidate < n1 else n1
            if candidate >= n1:
                continue
            # Get the suffix match (if any) for positions k+1 ... n2-1.
            suffix_seq = get_exact_seq(candidate+1, k+1) if k < n2-1 else tuple()
            if suffix_seq is None:
                continue
            candidate_seq = prefix_seq + (candidate,) + suffix_seq
            if len(candidate_seq) != n2:
                continue
            if best_seq is None or candidate_seq < best_seq:
                best_seq = candidate_seq
        
        return list(best_seq) if best_seq is not None else []

# --- Simple testing ---
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    word1 = "vbcca"
    word2 = "abc"
    print(sol.validSequence(word1, word2))  # Expected output: [0, 1, 2]

    # Example 2:
    word1 = "bacdc"
    word2 = "abc"
    print(sol.validSequence(word1, word2))  # Expected output: [1, 2, 4]

    # Example 3:
    word1 = "aaaaaa"
    word2 = "aaabc"
    print(sol.validSequence(word1, word2))  # Expected output: []

    # Example 4:
    word1 = "abc"
    word2 = "ab"
    print(sol.validSequence(word1, word2))  # Expected output: [0, 1]