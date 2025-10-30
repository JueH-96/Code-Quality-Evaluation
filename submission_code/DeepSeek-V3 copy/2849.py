class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        for i in range(n):
            unique = set()
            imbalance = 0
            for j in range(i, n):
                if nums[j] not in unique:
                    unique.add(nums[j])
                    # Check the left neighbor
                    if nums[j] - 1 in unique:
                        imbalance -= 1
                    # Check the right neighbor
                    if nums[j] + 1 in unique:
                        imbalance -= 1
                    # Update imbalance
                    if len(unique) > 1:
                        imbalance += 1
                total += imbalance
        return total