class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        len_word1 = len(word1)
        len_word2 = len(word2)
        if len_word2 == 0 or len_word1 < len_word2:
            return 0
        
        # Precompute the required counts for each character in word2
        from collections import defaultdict
        required = defaultdict(int)
        for c in word2:
            required[c] += 1
        
        # Precompute the prefix sums for each character in word1
        prefix = [[0] * 26]
        for c in word1:
            new_counts = prefix[-1].copy()
            idx = ord(c) - ord('a')
            new_counts[idx] += 1
            prefix.append(new_counts)
        
        # Check if any required character's total count in word1 is insufficient
        total_in_word1 = defaultdict(int)
        for c in word1:
            total_in_word1[c] += 1
        for c in required:
            if total_in_word1.get(c, 0) < required[c]:
                return 0
        
        result = 0
        
        # For each left, find the minimal right using binary search
        for left in range(len_word1):
            low = left + len_word2 - 1
            if low >= len_word1:
                continue  # Not enough characters to form the substring
            high = len_word1 - 1
            minimal_r = -1
            while low <= high:
                mid = (low + high) // 2
                # Check if the window [left, mid] satisfies the condition
                satisfied = True
                for c in required:
                    cnt = prefix[mid + 1][ord(c) - ord('a')] - prefix[left][ord(c) - ord('a')]
                    if cnt < required[c]:
                        satisfied = False
                        break
                if satisfied:
                    minimal_r = mid
                    high = mid - 1  # Try to find a smaller right
                else:
                    low = mid + 1  # Need to increase right
            if minimal_r != -1:
                result += (len_word1 - minimal_r)
        
        return result