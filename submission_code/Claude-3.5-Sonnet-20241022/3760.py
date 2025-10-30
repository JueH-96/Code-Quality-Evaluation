class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        n = len(groups)
        result = []
        
        # For each group
        for group_size in groups:
            assigned = -1
            # Try each element
            for j in range(len(elements)):
                # Check if group_size is divisible by current element
                if elements[j] != 0 and group_size % elements[j] == 0:
                    # Found a valid element, use its index
                    assigned = j
                    break
            result.append(assigned)
            
        return result