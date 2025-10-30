class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def get_xor_score(arr):
            n = len(arr)
            if n == 1:
                return arr[0]
            
            temp = arr.copy()
            while len(temp) > 1:
                new_temp = []
                for i in range(len(temp)-1):
                    new_temp.append(temp[i] ^ temp[i+1])
                temp = new_temp
            return temp[0]
        
        def get_max_xor_score(nums, left, right):
            max_score = 0
            n = right - left + 1
            
            # Try all possible subarrays
            for i in range(left, right + 1):
                for j in range(i, right + 1):
                    subarray = nums[i:j+1]
                    score = get_xor_score(subarray)
                    max_score = max(max_score, score)
                    
            return max_score
        
        answer = []
        for l, r in queries:
            max_score = get_max_xor_score(nums, l, r)
            answer.append(max_score)
            
        return answer