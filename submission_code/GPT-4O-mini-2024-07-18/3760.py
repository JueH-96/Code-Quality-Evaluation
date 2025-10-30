from typing import List

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        assigned = [-1] * len(groups)  # Initialize the assigned list with -1
        
        for i in range(len(groups)):
            for j in range(len(elements)):
                if groups[i] % elements[j] == 0:  # Check if the group size is divisible by the element
                    assigned[i] = j  # Assign the index of the element
                    break  # Stop checking further elements once we find a valid one
        
        return assigned