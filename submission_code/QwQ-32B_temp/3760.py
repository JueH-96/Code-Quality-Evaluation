class Solution:
    def __init__(self):
        max_g = 10**5
        self.divisors = [[] for _ in range(max_g + 1)]
        for i in range(1, max_g + 1):
            for j in range(i, max_g + 1, i):
                self.divisors[j].append(i)
    
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        # Preprocess elements to find the earliest occurrence of each value
        earliest = {}
        for idx, v in enumerate(elements):
            if v not in earliest:
                earliest[v] = idx
        
        result = []
        for g in groups:
            min_j = float('inf')
            divs = self.divisors[g]
            for d in divs:
                if d in earliest:
                    e = earliest[d]
                    if e < min_j:
                        min_j = e
                        if e == 0:  # Early exit if we found the smallest possible index
                            break
            if min_j == float('inf'):
                result.append(-1)
            else:
                result.append(min_j)
        return result