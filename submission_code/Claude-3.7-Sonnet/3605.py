class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        
        for x in nums:
            # Check if LSB is 0, if so there can't be a valid solution
            if x & 1 == 0:
                ans.append(-1)
                continue
                
            # Check if there's a 0 with all 1s to its right
            # This is another condition that makes a solution impossible
            binary_x = bin(x)[2:]
            impossible = False
            
            for i in range(len(binary_x) - 2, -1, -1):
                if binary_x[i] == '0':
                    all_ones_to_right = True
                    for j in range(i + 1, len(binary_x)):
                        if binary_x[j] == '0':
                            all_ones_to_right = False
                            break
                    if all_ones_to_right:
                        impossible = True
                        break
            
            if impossible:
                ans.append(-1)
                continue
                
            # If we get here, a solution exists
            # Find the smallest a where a | (a + 1) = x
            found = False
            for a in range(1001):  # Since nums[i] <= 1000
                if a | (a + 1) == x:
                    ans.append(a)
                    found = True
                    break
                    
            if not found:
                ans.append(-1)
                
        return ans