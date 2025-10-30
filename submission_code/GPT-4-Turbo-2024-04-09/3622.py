class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        from collections import Counter
        from sortedcontainers import SortedList
        
        if numOperations == 0:
            return max(Counter(nums).values())
        
        nums.sort()
        freq = Counter(nums)
        max_freq = max(freq.values())
        sorted_keys = SortedList(freq.keys())
        
        for _ in range(numOperations):
            # Try to maximize the frequency of the most frequent element
            # by adjusting nearby elements within the range of k
            for key in sorted_keys:
                # Check the next element in the sorted list
                idx = sorted_keys.bisect_right(key)
                if idx < len(sorted_keys):
                    next_key = sorted_keys[idx]
                    if next_key - key <= k:
                        # Move one occurrence of next_key to key
                        freq[key] += 1
                        freq[next_key] -= 1
                        if freq[next_key] == 0:
                            sorted_keys.remove(next_key)
                        max_freq = max(max_freq, freq[key])
                        break
                # Check the previous element in the sorted list
                if idx > 0:
                    prev_key = sorted_keys[idx - 1]
                    if key - prev_key <= k:
                        # Move one occurrence of key to prev_key
                        freq[prev_key] += 1
                        freq[key] -= 1
                        if freq[key] == 0:
                            sorted_keys.remove(key)
                        max_freq = max(max_freq, freq[prev_key])
                        break
        
        return max_freq