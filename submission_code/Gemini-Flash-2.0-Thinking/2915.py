class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        n = len(nums)
        modified_nums = [1 if num % modulo == k else 0 for num in nums]
        prefix_modified = [0] * (n + 1)
        for i in range(n):
            prefix_modified[i + 1] = prefix_modified[i] + modified_nums[i]

        count = 0
        mod_counts = {0: 1}  # Stores counts of prefix_modified[l] % modulo

        for r in range(n):
            target_mod = (prefix_modified[r + 1] - k) % modulo
            if target_mod < 0:
                target_mod += modulo

            count += mod_counts.get(target_mod, 0)

            current_mod = prefix_modified[r + 1] % modulo
            mod_counts[current_mod] = mod_counts.get(current_mod, 0) + 1

        return count