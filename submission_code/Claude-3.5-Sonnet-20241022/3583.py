class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        n = len(nums)
        gcd_pairs = []
        
        # Calculate GCD for all pairs
        for i in range(n):
            for j in range(i + 1, n):
                gcd_pairs.append(gcd(nums[i], nums[j]))
                
        # Sort GCD pairs in ascending order
        gcd_pairs.sort()
        
        # Get answer for each query
        answer = []
        for q in queries:
            answer.append(gcd_pairs[q])
            
        return answer