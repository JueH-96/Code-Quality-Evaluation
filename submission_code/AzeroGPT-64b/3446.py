class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        
        # counting frequency of each number in nums2 multplied by k
        freq = Counter(i * k for i in nums2)
        
        # initializing the result to 0
        result = 0
        
        # iterating over nums1 and for each number in nums1,
        # if the number is divisble by any key of freq dictionary, increment res by freq[key]
        for i in nums1:
            result += sum(freq[j] for j in freq if i % j == 0)
            
        return result