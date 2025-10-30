from collections import Counter

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        unique_nums = set(nums1) | set(nums2)
        unique_nums_list = sorted(list(unique_nums))
        max_set_size = 0
        
        from itertools import product
        
        counts1 = Counter(nums1)
        counts2 = Counter(nums2)
        
        best_set_size = 0
        
        import itertools
        
        indices1 = list(range(n))
        indices2 = list(range(n))
        
        for removed_indices1_tuple in itertools.combinations(indices1, n // 2):
            removed_indices1 = set(removed_indices1_tuple)
            remaining_nums1 = []
            for i in indices1:
                if i not in removed_indices1:
                    remaining_nums1.append(nums1[i])
                    
            for removed_indices2_tuple in itertools.combinations(indices2, n // 2):
                removed_indices2 = set(removed_indices2_tuple)
                remaining_nums2 = []
                for i in indices2:
                    if i not in removed_indices2:
                        remaining_nums2.append(nums2[i])
                        
                current_set = set(remaining_nums1) | set(remaining_nums2)
                best_set_size = max(best_set_size, len(current_set))
                
        return best_set_size