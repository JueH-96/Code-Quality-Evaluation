from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        total_sum = sum(nums)
        marked = [False] * n
        
        # Pre-sort the array by (value, index)
        sorted_list = sorted([(nums[i], i) for i in range(n)])
        pointer = 0  # pointer for the sorted_list
        
        answer = []
        
        for idx, k in queries:
            # Mark the element at index idx if not already marked.
            if not marked[idx]:
                marked[idx] = True
                total_sum -= nums[idx]
            
            # Mark k smallest unmarked elements using sorted_list approach.
            count = 0
            while count < k and pointer < n:
                value, pos = sorted_list[pointer]
                pointer += 1  # move pointer anyway
                if not marked[pos]:
                    marked[pos] = True
                    total_sum -= value
                    count += 1
            
            answer.append(total_sum)
        
        return answer

# Example usage:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    nums1 = [1,2,2,1,2,3,1]
    queries1 = [[1,2],[3,3],[4,2]]
    print(sol.unmarkedSumArray(nums1, queries1))  # Expected output: [8,3,0]
    
    # Example 2:
    nums2 = [1,4,2,3]
    queries2 = [[0,1]]
    print(sol.unmarkedSumArray(nums2, queries2))  # Expected output: [7]