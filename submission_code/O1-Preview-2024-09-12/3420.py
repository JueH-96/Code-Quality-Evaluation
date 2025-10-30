class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        positions = []
        for i, num in enumerate(nums):
            if num == x:
                positions.append(i)
        ans = []
        for q in queries:
            if q <= len(positions):
                ans.append(positions[q-1])
            else:
                ans.append(-1)
        return ans