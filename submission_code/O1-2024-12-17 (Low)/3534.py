class Solution:
    def countPairs(self, nums: List[int]) -> int:
        
        def can_be_equal_with_one_swap(x: int, y: int) -> bool:
            # If already equal, no swap needed.
            if x == y:
                return True
            
            s = str(x)
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    # Swap characters at positions i and j
                    arr = list(s)
                    arr[i], arr[j] = arr[j], arr[i]
                    
                    # Convert to integer (leading zeros are automatically dropped)
                    if int("".join(arr)) == y:
                        return True
            return False
        
        def almost_equal(a: int, b: int) -> bool:
            # They are almost equal if a -> b by one swap OR b -> a by one swap
            return can_be_equal_with_one_swap(a, b) or can_be_equal_with_one_swap(b, a)
        
        count = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i+1, n):
                if almost_equal(nums[i], nums[j]):
                    count += 1
        
        return count