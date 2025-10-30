from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        answer = 0
        
        # For each possible left index, we build subarrays starting at "left"
        # and extend them to the right.
        for left in range(n):
            # Frequency array: freq[x] tracks how many times x appears
            # We'll pad the array to avoid boundary issues with x-1 and x+1
            freq = [0] * (n + 2)  
            
            distinct_count = 0    # Number of distinct elements in the current subarray
            adjacency_count = 0   # Number of pairs (k, k+1) both present in the subarray
            
            for right in range(left, n):
                x = nums[right]
                if freq[x] == 0:
                    # When we see a new distinct element
                    freq[x] = 1
                    distinct_count += 1
                    
                    # If x-1 is present, we form a new adjacency pair (x-1, x)
                    if freq[x - 1] > 0:
                        adjacency_count += 1
                    # If x+1 is present, we form a new adjacency pair (x, x+1)
                    if freq[x + 1] > 0:
                        adjacency_count += 1
                else:
                    # Element already present, just update its frequency
                    freq[x] += 1
                
                # Imbalance calculation:
                # If there are 0 or 1 distinct elements, imbalance is 0.
                # Otherwise, it's (distinct_count - 1) minus the number of adjacent pairs
                if distinct_count <= 1:
                    imbalance = 0
                else:
                    imbalance = (distinct_count - 1) - adjacency_count
                
                answer += imbalance
        
        return answer