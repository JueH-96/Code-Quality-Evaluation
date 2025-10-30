class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        distinct_nums = set(nums)
        num_distinct = len(distinct_nums)
        
        duplicates = []
        for num, count in counts.items():
            if count > 1:
                duplicates.append((num, count))
        
        duplicates.sort()
        
        
        i = 0
        while i < len(duplicates):
            num, count = duplicates[i]
            
            merged = False
            for j in range(i + 1, len(duplicates)):
                next_num, next_count = duplicates[j]
                if abs(num - next_num) <= 2 * k:
                    
                    k_needed = abs(num - next_num)
                    if k_needed <= 2 *k:
                        k -= k_needed //2
                        if k >=0:
                            num_distinct -=1
                            duplicates[j] = (next_num,0)
                            merged = True
                            break
                        else:
                            break
            if not merged:
                i+=1

        return num_distinct