class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        # Create a dictionary to map each element value to the smallest index it appears at
        element_indices = {}
        for idx, val in enumerate(elements):
            if val not in element_indices:
                element_indices[val] = idx
        
        assigned = []
        for group in groups:
            # Find all divisors of group
            divisors = set()
            for i in range(1, int(group ** 0.5) + 1):
                if group % i == 0:
                    divisors.add(i)
                    divisors.add(group // i)
            # Check each divisor in elements, track the smallest index
            min_index = float('inf')
            for d in divisors:
                if d in element_indices and element_indices[d] < min_index:
                    min_index = element_indices[d]
            if min_index != float('inf'):
                assigned.append(min_index)
            else:
                assigned.append(-1)
        return assigned