class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        assigned = []
        
        for group in groups:
            found = False
            for j, element in enumerate(elements):
                if group % element == 0:
                    assigned.append(j)
                    found = True
                    break
            if not found:
                assigned.append(-1)
        
        return assigned