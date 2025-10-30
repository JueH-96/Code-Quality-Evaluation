class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        result = []
        
        for group_size in groups:
            for j, element in enumerate(elements):
                if group_size % element == 0:
                    result.append(j)
                    break
            else:
                result.append(-1)
        
        return result