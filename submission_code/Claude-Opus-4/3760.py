class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        result = []
        
        for group_size in groups:
            assigned_index = -1
            
            # Check each element in order (smallest index first)
            for j in range(len(elements)):
                if group_size % elements[j] == 0:
                    assigned_index = j
                    break
            
            result.append(assigned_index)
        
        return result