from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        # We will use an incremental approach, tracking in O(1) how the "imbalance" of 
        # each subarray [start..end] changes as end moves from start to n-1.
        #
        # Observing that the imbalance is the count of "gaps" in the sorted distinct elements
        # of the subarray, we can track the number of distinct elements (d) and the number of
        # pairs of distinct elements that differ by exactly one (c). Then:
        #
        #   number_of_pairs = d - 1
        #   number_of_pairs_that_are_not_gaps = c   (pairs x, x+1 both present)
        #   imbalance = (d - 1) - c
        #
        # We maintain a freq array (count of occurrences) and a present boolean array to
        # quickly check if a value is in the subarray. Whenever we add a new distinct value x,
        # we increase d by 1 and update c based on neighbors (x-1 and x+1).
        
        result = 0
        
        for start in range(n):
            freq = [0]*(n+1)      # frequency of each possible value (1..n)
            present = [False]*(n+1)
            d = 0  # number of distinct elements
            c = 0  # count of adjacent pairs (x, x+1) that are both in the set
            
            for end in range(start, n):
                x = nums[end]
                if freq[x] == 0:
                    # x is a new distinct element in this subarray
                    d += 1
                    present[x] = True
                    # check neighbor x-1
                    if x > 1 and present[x-1]:
                        c += 1
                    # check neighbor x+1
                    if x < n and present[x+1]:
                        c += 1
                freq[x] += 1
                
                # if we have d distinct elements,
                # imbalance = (d - 1) - c
                if d > 0:
                    imbalance = (d - 1) - c
                else:
                    imbalance = 0
                
                result += imbalance
        
        return result