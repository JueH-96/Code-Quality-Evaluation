class Solution:
    def countPairs(self, nums: List[int]) -> int:
        # Helper function to generate all integers obtainable
        # by swapping at most two digits in x (including x itself).
        def generate_swapped_values(x: int) -> set:
            s = str(x)
            values = {x}  # include x itself (no swap)
            arr = list(s)
            n = len(arr)
            for i in range(n):
                for j in range(i + 1, n):
                    arr[i], arr[j] = arr[j], arr[i]
                    # Convert back to int to remove any leading zeros
                    values.add(int("".join(arr)))
                    # Swap back to restore original string for next iteration
                    arr[i], arr[j] = arr[j], arr[i]
            return values
        
        ans = 0
        n = len(nums)
        
        # Precompute the sets of swapped values for each number
        # to speed up lookup for repeated numbers.
        swapped_map = {}
        for x in nums:
            if x not in swapped_map:
                swapped_map[x] = generate_swapped_values(x)
        
        # Count pairs (i, j) with i < j that are "almost equal"
        for i in range(n):
            for j in range(i + 1, n):
                x, y = nums[i], nums[j]
                # Check if, with at most one swap, x can become y or y can become x
                if (y in swapped_map[x]) or (x in swapped_map.setdefault(y, generate_swapped_values(y))):
                    ans += 1
        
        return ans