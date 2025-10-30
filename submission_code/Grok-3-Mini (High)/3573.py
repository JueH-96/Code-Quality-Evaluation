import bisect

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        # Compute required counts for each character in word2
        req_count = [0] * 26
        for char in word2:
            idx = ord(char) - ord('a')
            req_count[idx] += 1
        
        # Find indices of characters that have required count > 0
        required_indices = [i for i in range(26) if req_count[i] > 0]
        num_required = len(required_indices)
        
        n = len(word1)
        
        # Compute prefix sums for each required character
        prefix_lists = []
        for c_idx in required_indices:
            prefix_c = [0] * (n + 1)  # prefix_c[i] = count of char up to index i-1
            for i in range(1, n + 1):
                if ord(word1[i - 1]) - ord('a') == c_idx:
                    prefix_c[i] = prefix_c[i - 1] + 1
                else:
                    prefix_c[i] = prefix_c[i - 1]
            prefix_lists.append(prefix_c)
        
        # Now, for each right endpoint, compute the number of valid left endpoints
        ans = 0
        for right in range(n):  # right from 0 to n-1
            # Compute min_L
            min_L = None
            for k in range(num_required):
                c_idx = required_indices[k]
                prefix_c_list = prefix_lists[k]
                val_c = prefix_c_list[right + 1] - req_count[c_idx]
                pos = bisect.bisect_right(prefix_c_list, val_c)
                L_c = pos - 1
                if min_L is None or L_c < min_L:
                    min_L = L_c
            # After finding min_L
            if min_L >= 0:
                max_left_index = min(right, min_L)
                ans += max_left_index + 1
        
        return ans