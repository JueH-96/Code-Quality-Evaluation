from typing import List

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n - k + 1):
            sub = nums[i:i+k]
            freq = {}
            for num in sub:
                if num in freq:
                    freq[num] += 1
                else:
                    freq[num] = 1
            elements = list(freq.items())
            elements.sort(key=lambda e: (-e[1], -e[0]))
            selected = set()
            for j in range(min(x, len(elements))):
                selected.add(elements[j][0])
            sum_x = sum(num for num in sub if num in selected)
            result.append(sum_x)
        return result