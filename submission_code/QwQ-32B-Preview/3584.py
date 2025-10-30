from typing import List
import bisect

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        # Precompute positions for each character in word2
        pos = {ch: [] for ch in set(word2)}
        for idx, ch in enumerate(word1):
            if ch in pos:
                pos[ch].append(idx)
        
        n = len(word2)
        INF = float('inf')
        exact = [INF] * n
        mismatch = [INF] * n
        
        # Initialize for the first character
        if word2[0] in pos:
            exact[0] = pos[word2[0]][0] if pos[word2[0]] else INF
        mismatch[0] = min(pos[ch][0] for ch in pos if ch != word2[0]) if any(ch != word2[0] for ch in pos) else INF
        
        # Fill the dp arrays
        for j in range(1, n):
            if exact[j-1] < INF:
                # Find the next index for exact match
                idx_list = pos[word2[j]]
                idx = bisect.bisect_left(idx_list, exact[j-1] + 1)
                if idx < len(idx_list):
                    exact[j] = idx_list[idx]
                # Find the next index for mismatch
                for ch in pos:
                    if ch != word2[j]:
                        idx_list = pos[ch]
                        idx = bisect.bisect_left(idx_list, exact[j-1] + 1)
                        if idx < len(idx_list):
                            mismatch[j] = min(mismatch[j], idx_list[idx])
            
            if mismatch[j-1] < INF:
                # Find the next index for exact match with previous mismatch
                idx_list = pos[word2[j]]
                idx = bisect.bisect_left(idx_list, mismatch[j-1] + 1)
                if idx < len(idx_list):
                    mismatch[j] = min(mismatch[j], idx_list[idx])
                # Find the next index for mismatch with previous mismatch
                for ch in pos:
                    if ch != word2[j]:
                        idx_list = pos[ch]
                        idx = bisect.bisect_left(idx_list, mismatch[j-1] + 1)
                        if idx < len(idx_list):
                            mismatch[j] = min(mismatch[j], idx_list[idx])
        
        # Reconstruct the sequence if possible
        if exact[-1] < INF:
            return self.reconstruct(word1, word2, exact, True)
        elif mismatch[-1] < INF:
            return self.reconstruct(word1, word2, mismatch, False)
        else:
            return []
    
    def reconstruct(self, word1, word2, dp, is_exact):
        seq = []
        index = dp[-1]
        seq.append(index)
        j = len(word2) - 1
        while j > 0:
            prev_index = dp[j-1]
            if is_exact:
                if word1[index] == word2[j]:
                    index = prev_index
                    seq.append(index)
                    j -= 1
                else:
                    is_exact = False
            else:
                if word1[index] != word2[j]:
                    index = prev_index
                    seq.append(index)
                    j -= 1
                else:
                    # Try to find the exact match path
                    potential_index = bisect.bisect_left(dp, index, 0, j)
                    if potential_index < j and dp[potential_index] < index:
                        index = dp[potential_index]
                        seq.append(index)
                        j = potential_index
                        is_exact = True
                    else:
                        index = prev_index
                        seq.append(index)
                        j -= 1
        seq.reverse()
        return seq