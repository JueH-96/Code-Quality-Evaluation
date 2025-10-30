class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        ans = 0
        
        def combinations(arr, r):
            result = []
            
            def generate(index, current):
                if len(current) == r:
                    result.append(current.copy())
                    return
                
                if index >= len(arr):
                    return
                
                generate(index + 1, current)
                current.append(arr[index])
                generate(index + 1, current)
            
            generate(0, [])
            return result

        
        for combo in combinations(nums,k):
            min_diff = float('inf')
            for i in range(len(combo)):
                for j in range(i+1,len(combo)):
                    min_diff = min(min_diff, abs(combo[i]-combo[j]))
            ans = (ans + min_diff)%MOD

        return ans