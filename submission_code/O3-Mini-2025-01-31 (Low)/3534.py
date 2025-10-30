class Solution:
    def countPairs(self, nums: List[int]) -> int:
        # A helper function to generate all numbers reachable from the given number with at most one swap.
        # Represent the number as a string (its digits). A swap operation is allowed only on one of the two numbers.
        # When swapping digits, leading zeros are allowed.
        def reachable(num_str: str) -> set:
            # We'll include the original number and all numbers obtained by exactly one swap.
            s = list(num_str)
            n = len(s)
            res = {int(num_str)}
            # Try swapping any two distinct positions i and j (i < j)
            for i in range(n):
                for j in range(i+1, n):
                    # Copy list to avoid modifying original list.
                    arr = s[:]
                    arr[i], arr[j] = arr[j], arr[i]
                    # Convert to integer, allowing leading zeros by direct conversion.
                    res.add(int("".join(arr)))
            return res

        # Memorization dictionary to avoid recalculating reachable swaps for identical number-string representations.
        memo = {}
        def get_reachable(num: int) -> set:
            num_str = str(num)
            if num_str not in memo:
                memo[num_str] = reachable(num_str)
            return memo[num_str]

        count = 0
        n = len(nums)
        for i in range(n):
            rep_i = get_reachable(nums[i])
            for j in range(i+1, n):
                # Check if swapping in nums[i] can yield nums[j]
                if nums[j] in rep_i:
                    count += 1
                else:
                    # Otherwise, check if swapping in nums[j] can yield nums[i]
                    rep_j = get_reachable(nums[j])
                    if nums[i] in rep_j:
                        count += 1
        return count