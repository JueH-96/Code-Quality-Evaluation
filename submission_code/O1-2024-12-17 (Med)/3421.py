class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        count = 0
        n = len(hours)
        
        # Since hours.length <= 100, we can use a simple O(n^2) approach
        for i in range(n):
            for j in range(i+1, n):
                if (hours[i] + hours[j]) % 24 == 0:
                    count += 1
        
        return count