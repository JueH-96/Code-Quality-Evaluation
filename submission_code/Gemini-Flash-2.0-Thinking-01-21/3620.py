class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        duplicates_count = 0
        for num in counts:
            if counts[num] > 1:
                duplicates_count += counts[num] - 1
                
        distinct_elements = set(counts.keys())
        operations_used = 0
        
        sorted_counts_items = sorted(counts.items())
        
        for num, freq in sorted_counts_items:
            if freq > 1:
                duplicates_to_remove = freq - 1
                for _ in range(duplicates_to_remove):
                    if operations_used >= k:
                        return len(distinct_elements)
                    original_num = num
                    changed = False
                    for delta in range(1, k + 1):
                        next_num_plus = original_num + delta
                        if next_num_plus not in distinct_elements:
                            distinct_elements.add(next_num_plus)
                            operations_used += 1
                            changed = True
                            break
                        next_num_minus = original_num - delta
                        if next_num_minus not in distinct_elements:
                            distinct_elements.add(next_num_minus)
                            operations_used += 1
                            changed = True
                            break
                    if not changed:
                        next_num = original_num + k + 1
                        distinct_elements.add(next_num)
                        operations_used += 1
                        
        return len(distinct_elements)