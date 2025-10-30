class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        mod = 10 ** 9 + 7
        
        # Create a frequency map of the elements in the array
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        # Define a recursive function to generate all possible sub-multisets
        def generate_subsets(start, curr_sum):
            if curr_sum > r:
                return 0
            
            count = 0
            if l <= curr_sum <= r:
                count += 1
            
            for i in range(start, n):
                num = nums[i]
                for j in range(1, freq[num] + 1):
                    count += generate_subsets(i + 1, curr_sum + num * j)
                    count %= mod
            
            return count
        
        return generate_subsets(0, 0)