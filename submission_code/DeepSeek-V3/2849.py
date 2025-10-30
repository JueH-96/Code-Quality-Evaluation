class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        for i in range(n):
            unique_elements = set()
            imbalance = 0
            for j in range(i, n):
                if nums[j] not in unique_elements:
                    unique_elements.add(nums[j])
                    # Check the left and right neighbors
                    left = nums[j] - 1
                    right = nums[j] + 1
                    # Count the number of existing elements that are left and right
                    count_left = 1 if left in unique_elements else 0
                    count_right = 1 if right in unique_elements else 0
                    # Update imbalance based on the counts
                    if count_left == 0 and count_right == 0:
                        imbalance += 1
                    elif count_left == 1 and count_right == 1:
                        imbalance -= 1
                total += imbalance
        return total