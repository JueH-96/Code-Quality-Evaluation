class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        index_of_1 = nums.index(1)
        index_of_n = nums.index(n)

        # Move 1 to the front
        steps_to_move_1 = index_of_1

        # Move n to the end
        if index_of_1 < index_of_n:
            steps_to_move_n = index_of_n - 1
        else:
            steps_to_move_n = index_of_n

        # Total steps
        total_steps = steps_to_move_1 + steps_to_move_n

        return total_steps