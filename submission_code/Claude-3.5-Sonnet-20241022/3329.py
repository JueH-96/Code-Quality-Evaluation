class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Convert all numbers to strings for easier prefix comparison
        arr1_str = [str(x) for x in arr1]
        arr2_str = [str(x) for x in arr2]
        
        max_len = 0
        
        # Compare each pair of numbers
        for num1 in arr1_str:
            for num2 in arr2_str:
                # Find length of shorter number
                min_len = min(len(num1), len(num2))
                
                # Find length of common prefix
                prefix_len = 0
                for i in range(min_len):
                    if num1[i] == num2[i]:
                        prefix_len += 1
                    else:
                        break
                        
                max_len = max(max_len, prefix_len)
                
        return max_len