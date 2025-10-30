from collections import defaultdict

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        target_counts = defaultdict(int)
        for x in nums:
            target_counts[x] += 1
            target_counts[x + 1] += 1
        targets = sorted(target_counts.keys())
        max_len = 0
        current_len = 0
        for i in range(len(targets)):
            if i == 0:
                current_len = 1
            else:
                if targets[i] == targets[i-1] + 1:
                    current_len += 1
                else:
                    current_len = 1
            if current_len > max_len:
                max_len = current_len
        return max_len