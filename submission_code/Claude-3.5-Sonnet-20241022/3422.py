class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        arr = [1] * n
        
        for _ in range(k):
            new_arr = [0] * n
            new_arr[0] = arr[0]
            
            for i in range(1, n):
                # Sum of all preceding elements plus current element
                new_arr[i] = (sum(arr[:i+1])) % MOD
                
            arr = new_arr
            
        return arr[n-1]