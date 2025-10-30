from bisect import bisect_left

class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        window_len = len(pattern)
        prefix_hash_pattern = [ord(pattern[i]) - 96 for i in range(window_len)]
        for i in range(len(prefix_hash_pattern) - 2, -1, -1):
            prefix_hash_pattern[i] += prefix_hash_pattern[i+1]

        hash_hash_s = [ord(c) - 96 for c in s]
        for i in range(len(hash_hash_s) - 1, len(s) - window_len, -1):
            hash_hash_s[i - window_len + 1] += hash_hash_s[i + 1] if i + 1 < len(hash_hash_s) else 0

        mismatch_indices = []
        for i in range(len(s) - window_len + 1):
            window_hash = hash_hash_s[i] if i + window_len < len(hash_hash_s) else sum(ord(c) - 96 for c in s[i:])
            if window_hash == prefix_hash_pattern[0]:
                mismatch = False
                for j in range(window_len):
                    if s[i+j] != pattern[j]:
                        mismatch_indices.append(i + j)
                        mismatch = True
                        break
                if not mismatch:
                    return i

        def can_change_only_one(index):
            left = index - 1
            right = index + 1
            while left >= 0 and right < len(mismatch_indices) and s[mismatch_indices[left]] == pattern[mismatch_indices[right] - mismatch_indices[left] - 1]:
                left -= 1
                right += 1
            return left == -1 or right == len(mismatch_indices)

        mismatch_indices.sort()
        for index in mismatch_indices:
            if can_change_only_one(index):
                return mismatch_indices[bisect_left(mismatch_indices, index - (index - 1) % window_len) - 1] - (index - 1) % window_len

        return -1