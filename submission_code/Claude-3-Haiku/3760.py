class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        assigned = [-1] * len(groups)
        elements_dict = {}
        
        # Create a dictionary to store the elements and their indices
        for i, element in enumerate(elements):
            if element not in elements_dict:
                elements_dict[element] = i
        
        # Assign elements to groups
        for i, group_size in enumerate(groups):
            for element, index in elements_dict.items():
                if group_size % element == 0:
                    assigned[i] = index
                    break
        
        return assigned