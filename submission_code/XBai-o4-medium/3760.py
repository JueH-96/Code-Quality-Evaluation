from typing import List

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        value_to_index = {}
        for idx, num in enumerate(elements):
            if num not in value_to_index:
                value_to_index[num] = idx
        
        def get_divisors(n: int):
            divisors = set()
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    divisors.add(i)
                    divisors.add(n // i)
            return divisors
        
        result = []
        for g in groups:
            divisors = get_divisors(g)
            candidates = []
            for d in divisors:
                if d in value_to_index:
                    candidates.append(value_to_index[d])
            if candidates:
                result.append(min(candidates))
            else:
                result.append(-1)
        return result