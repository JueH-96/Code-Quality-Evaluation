class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        result = []
        
        for group in groups:
            assigned = -1
            for j, element in enumerate(elements):
                if group % element == 0:
                    assigned = j
                    break
            result.append(assigned)
        
        return result