class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        assigned = []
        
        for group_size in groups:
            chosen_index = -1
            
            # Check each element in order (to get smallest index first)
            for j, element in enumerate(elements):
                if group_size % element == 0:
                    chosen_index = j
                    break  # Take the first (smallest index) valid element
            
            assigned.append(chosen_index)
        
        return assigned