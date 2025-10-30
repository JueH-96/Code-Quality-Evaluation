class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        result = []
        n = len(nums)
        for i in range(n - k + 1):
            sub = nums[i:i+k]
            freq = {}
            for num in sub:
                if num in freq:
                    freq[num] += 1
                else:
                    freq[num] = 1
            items = list(freq.items())
            items.sort(key=lambda item: (-item[1], -item[0]))
            top = items[:x]
            selected = {e for e, _ in top}
            total = sum(e for e in sub if e in selected)
            result.append(total)
        return result