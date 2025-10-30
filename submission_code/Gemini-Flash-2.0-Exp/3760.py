class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        assigned = []
        for group in groups:
            found = False
            for i, element in enumerate(elements):
                if group % element == 0:
                    assigned.append(i)
                    found = True
                    break
            if not found:
                assigned.append(-1)
        return assigned