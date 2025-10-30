class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        n = len(nums)
        arr = sorted([(num, bin(num).count('1')) for num in nums])
        
        for i in range(n):
            curr = nums[i]
            curr_bits = bin(curr).count('1')
            
            
            j = 0
            while j < n:
                
                k = j
                while k < n and bin(nums[k]).count('1') == curr_bits:
                    k += 1
                
                sub_array = nums[j:k]
                sub_array.sort()
                nums[j:k] = sub_array
                
                j = k
        
        
        for i in range(n - 1):
            if nums[i] > nums[i+1]:
                return False
        
        return True