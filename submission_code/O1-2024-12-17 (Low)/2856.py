class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        # Number of distinct elements in the entire array
        k = len(set(nums))
        
        # Function to count the number of subarrays with at most K distinct elements
        def atMostKDistinct(arr, K):
            left = 0
            freq = {}
            count = 0
            distinct_count = 0
            
            for right, val in enumerate(arr):
                freq[val] = freq.get(val, 0) + 1
                if freq[val] == 1:  # a new distinct element
                    distinct_count += 1
                    
                while distinct_count > K:
                    freq[arr[left]] -= 1
                    if freq[arr[left]] == 0:
                        del freq[arr[left]]
                        distinct_count -= 1
                    left += 1
                    
                count += (right - left + 1)
            return count
        
        # Number of subarrays with exactly k distinct = atMostKDistinct - atMost(K-1)
        return atMostKDistinct(nums, k) - atMostKDistinct(nums, k - 1)