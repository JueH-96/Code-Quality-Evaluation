class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        duplicates = []
        for num, count in counts.items():
            if count > 1:
                duplicates.append(count)
        
        duplicates.sort()
        
        distinct_count = len(counts)
        
        for count in duplicates:
            needed = count - 1
            if k >= needed:
                k -= needed
                distinct_count += 1
            else:
                break
        
        distinct_count -= len(counts) - len(set(nums))
        
        if k > 0:
            distinct_count = min(len(nums), distinct_count + min(len(counts) - len(duplicates), k))
        
        return distinct_count