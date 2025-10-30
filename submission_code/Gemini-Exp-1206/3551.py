class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def calculate_xor_score(arr):
            if not arr:
                return 0
            if len(arr) == 1:
                return arr[0]
            
            while len(arr) > 1:
                for i in range(len(arr) - 1):
                    arr[i] = arr[i] ^ arr[i+1]
                arr.pop()
            return arr[0]

        answer = []
        for l, r in queries:
            max_xor = 0
            sub_array = nums[l : r+1]
            for i in range(len(sub_array)):
                for j in range(i, len(sub_array)):
                    max_xor = max(max_xor, calculate_xor_score(sub_array[i : j+1]))
            answer.append(max_xor)
        return answer