from typing import List
import heapq
from collections import defaultdict

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        # Prepare array of (nums1[i], nums2[i], i)
        arr = [(nums1[i], nums2[i], i) for i in range(n)]
        # Sort by nums1 value; for equal nums1, order doesn't matter because they won't count for each other
        arr.sort(key=lambda x: x[0])
        
        # answer array, initialize all to 0
        ans = [0] * n
        # min-heap to maintain top k seen nums2 values from indices with smaller nums1
        heap = []
        curr_sum = 0
        
        # Process in groups of same nums1 value.
        i = 0
        while i < n:
            # group start with current nums1 value
            same_value_group = []
            curr_val = arr[i][0]
            while i < n and arr[i][0] == curr_val:
                same_value_group.append(arr[i])
                i += 1
                
            # For every element in the current group, the valid set is only those from strictly smaller nums1,
            # which have been processed. Answer is the current curr_sum.
            for _, _, idx in same_value_group:
                ans[idx] = curr_sum
                
            # Now add the current group candidates to the heap
            for _, candidate, _ in same_value_group:
                # if we have less than k, push it
                if len(heap) < k:
                    heapq.heappush(heap, candidate)
                    curr_sum += candidate
                else:
                    # if candidate is larger than the smallest in the heap, replace it to maximize sum
                    if heap and candidate > heap[0]:
                        removed = heapq.heappop(heap)
                        curr_sum -= removed
                        heapq.heappush(heap, candidate)
                        curr_sum += candidate
                    # otherwise, candidate is not used because including it would not increase the sum of top k
        return ans

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    nums1 = [4,2,1,5,3]
    nums2 = [10,20,30,40,50]
    k = 2
    print(sol.findMaxSum(nums1, nums2, k))  # Expected output: [80,30,0,80,50]

    # Example 2:
    nums1 = [2,2,2,2]
    nums2 = [3,1,2,3]
    k = 1
    print(sol.findMaxSum(nums1, nums2, k))  # Expected output: [0,0,0,0]