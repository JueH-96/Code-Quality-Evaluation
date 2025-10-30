from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []
        
        # Count the occurrences of all elements in the array
        freq = Counter(nums)
        
        # Sort the elements by their frequency in descending order, and then by their value in descending order
        sorted_freq = sorted(freq.items(), key=lambda x: (-x[1], -x[0]))
        
        # Keep only the occurrences of the top x most frequent elements
        top_x_elements = [x[0] for x in sorted_freq[:x]]
        
        # Calculate the x-sum of each subarray of length k
        for i in range(n - k + 1):
            subarray_sum = 0
            for num in nums[i:i+k]:
                if num in top_x_elements:
                    subarray_sum += num
            answer.append(subarray_sum)
        
        return answer