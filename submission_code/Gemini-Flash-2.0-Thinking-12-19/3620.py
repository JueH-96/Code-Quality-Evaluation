from collections import Counter

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        counts = Counter(nums)
        distinct_elements_set = set(nums)
        initial_distinct_count = len(distinct_elements_set)
        duplicates_nums = []
        for num in sorted(counts.keys()):
            if counts[num] > 1:
                duplicates_nums.extend([num] * (counts[num] - 1))
        
        increased_distinct_count = 0
        changed_values = set()
        
        for original_num in duplicates_nums:
            found_new_val = False
            for delta in range(1, k + 1):
                new_val_pos = original_num + delta
                new_val_neg = original_num - delta
                if new_val_pos not in distinct_elements_set and new_val_pos not in changed_values:
                    distinct_elements_set.add(new_val_pos)
                    changed_values.add(new_val_pos)
                    increased_distinct_count += 1
                    found_new_val = True
                    break
                elif new_val_neg not in distinct_elements_set and new_val_neg not in changed_values:
                    distinct_elements_set.add(new_val_neg)
                    changed_values.add(new_val_neg)
                    increased_distinct_count += 1
                    found_new_val = True
                    break
            if found_new_val:
                continue
                
        return initial_distinct_count + increased_distinct_count