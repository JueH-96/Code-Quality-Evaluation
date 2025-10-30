class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        # Initialize the result list with -1 for each group
        assigned = [-1] * len(groups)
        
        # Iterate over each group
        for i in range(len(groups)):
            # Iterate over each element to find the first suitable one
            for j in range(len(elements)):
                if groups[i] % elements[j] == 0:
                    assigned[i] = j
                    break  # Stop after finding the first suitable element
        
        return assigned