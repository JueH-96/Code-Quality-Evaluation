class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def get_all_swaps(num):
            s = str(num)
            swaps = {num}  # include the original number (no swap)
            n = len(s)
            for i in range(n):
                for j in range(i + 1, n):
                    # swap digits at positions i and j
                    chars = list(s)
                    chars[i], chars[j] = chars[j], chars[i]
                    swaps.add(int(''.join(chars)))
            return swaps
        
        # Precompute swaps for each unique number to avoid redundant computation
        swaps_cache = {}
        for num in set(nums):
            swaps_cache[num] = get_all_swaps(num)
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                # Check if nums[j] can be obtained from nums[i] by at most one swap
                # OR if nums[i] can be obtained from nums[j] by at most one swap
                if nums[j] in swaps_cache[nums[i]] or nums[i] in swaps_cache[nums[j]]:
                    count += 1
        return count