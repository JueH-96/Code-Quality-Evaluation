from typing import List

class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        for i in range(n):
            # Calculate the destination index in circular manner
            destination_index = (i + nums[i]) % n
            result[i] = nums[destination_index]
        return result

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.constructTransformedArray([3, -2, 1, 1]))  # Expected output: [1, 1, 1, 3]
    print(sol.constructTransformedArray([-1, 4, -1]))    # Expected output: [-1, -1, 4]