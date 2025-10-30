class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)

        # Create a list to track the number of times each index needs to be decremented
        decrement_count = [0] * n

        # Calculate the total number of decrements needed for each index
        for i in range(m):
            decrement_count[changeIndices[i] - 1] += 1

        # Create a list to track the number of times each index has been decremented
        decremented_count = [0] * n

        # Create a set to track the indices that have been marked
        marked_indices = set()

        # Iterate through each second
        for s in range(m):
            # Decrement the index specified by changeIndices[s]
            index = changeIndices[s] - 1
            if nums[index] > 0:
                nums[index] -= 1
                decremented_count[index] += 1

            # Check if the index can be marked
            if nums[index] == 0 and decremented_count[index] >= decrement_count[index]:
                marked_indices.add(index)

            # Check if all indices have been marked
            if len(marked_indices) == n:
                return s + 1

        # If not all indices can be marked, return -1
        return -1