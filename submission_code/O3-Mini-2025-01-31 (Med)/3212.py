class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        
        # For each number, record its last occurrence in the array.
        last_occurrence = {}
        for i, num in enumerate(nums):
            last_occurrence[num] = i
        
        # We iterate through the array and maintain a running maximum (curMax)
        # representing the farthest last occurrence for numbers seen so far.
        # A valid partition boundary (cut) can be placed between indices i and i+1
        # if i equals this curMax. This guarantees that all occurrences of any
        # number in the prefix (nums[0..i]) are contained in that prefix,
        # so the same number does not appear in the subsequent segment.
        valid_cuts = 0
        curMax = 0
        for i in range(n - 1):  # we do not consider the last index because it's automatically the end.
            curMax = max(curMax, last_occurrence[nums[i]])
            if i == curMax:
                valid_cuts += 1
        
        # For each valid cut, we have two options: either make a cut or not.
        # Thus, the total number of valid partitions is 2^(number of valid cut positions).
        return pow(2, valid_cuts, mod)
        
# --- Additional code for local running/testing (optional) ---
if __name__ == '__main__':
    # Example test cases
    sol = Solution()
    
    # Example 1
    nums1 = [1, 2, 3, 4]
    print(sol.numberOfGoodPartitions(nums1))  # Expected output: 8

    # Example 2
    nums2 = [1, 1, 1, 1]
    print(sol.numberOfGoodPartitions(nums2))  # Expected output: 1

    # Example 3
    nums3 = [1, 2, 1, 3]
    print(sol.numberOfGoodPartitions(nums3))  # Expected output: 2