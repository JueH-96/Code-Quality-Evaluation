class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        # Find all indices where x occurs in nums
        occurrences = [i for i, num in enumerate(nums) if num == x]
        
        # Process each query
        return [occurrences[q - 1] if q <= len(occurrences) else -1 for q in queries]