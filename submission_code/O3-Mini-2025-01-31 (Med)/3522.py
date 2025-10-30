from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        result = []
        # For each subarray of length k
        for i in range(n - k + 1):
            sub = nums[i:i+k]
            valid = True
            # For k==1, automatically valid, else check consecutive ascending order
            for j in range(1, k):
                if sub[j] - sub[j-1] != 1:
                    valid = False
                    break
            if valid:
                # Maximum element in a sorted ascending array is the last element
                result.append(sub[-1])
            else:
                result.append(-1)
        return result

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.resultsArray([1,2,3,4,3,2,5], 3))  # Output: [3,4,-1,-1,-1]
    print(sol.resultsArray([2,2,2,2,2], 4))       # Output: [-1,-1]
    print(sol.resultsArray([3,2,3,2,3,2], 2))       # Output: [-1,3,-1,3,-1]