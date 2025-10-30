class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        from collections import Counter
        
        if k == 0:
            return len(set(nums))
        
        count = Counter(nums)
        freq = sorted(count.items(), key=lambda x: x[1], reverse=True)
        
        used = set()
        changes = []
        
        for num, cnt in freq:
            if cnt == 1:
                if num not in used:
                    used.add(num)
                else:
                    changes.append(num)
            else:
                for i in range(cnt):
                    if num not in used:
                        used.add(num)
                    else:
                        changes.append(num)
        
        # Attempt to make each change unique if possible
        changes.sort()
        for change in changes:
            # Try to find a unique number by adjusting within [-k, k]
            for delta in range(-k, k + 1):
                new_value = change + delta
                if new_value not in used:
                    used.add(new_value)
                    break
        
        return len(used)