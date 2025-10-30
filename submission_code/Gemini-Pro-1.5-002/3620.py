class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        count = Counter(nums)
        duplicates = []
        for num, freq in count.items():
            if freq > 1:
                duplicates.append((num, freq))
        
        duplicates.sort()
        
        distinct_count = len(count)
        
        for num, freq in duplicates:
            diff = (freq - 1) * num
            
            if diff <= k:
                k -= diff
                distinct_count += freq -1
            else:
                distinct_count += k // num
                k = 0
                break
        return distinct_count