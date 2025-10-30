class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        used = set()
        
        for num in nums:
            if num not in used:
                used.add(num)
            else:
                # Try to find the smallest unused value within the valid range
                for val in range(max(num - k, 1), num + k + 1):
                    if val not in used:
                        used.add(val)
                        break
                # If no unused value is found, we skip this element
        
        return len(used)