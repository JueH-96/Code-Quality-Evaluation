class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # count[x] = number of subsequences ending with x
        # total[x] = sum of all elements in subsequences ending with x
        count = {}
        total = {}
        
        for num in nums:
            # Current number forms a new subsequence of length 1
            new_count = 1
            new_total = num
            
            # Extend subsequences ending with num-1
            if num - 1 in count:
                new_count = (new_count + count[num - 1]) % MOD
                new_total = (new_total + total[num - 1] + count[num - 1] * num) % MOD
            
            # Extend subsequences ending with num+1
            if num + 1 in count:
                new_count = (new_count + count[num + 1]) % MOD
                new_total = (new_total + total[num + 1] + count[num + 1] * num) % MOD
            
            count[num] = new_count
            total[num] = new_total
        
        # Sum all totals
        result = 0
        for val in total.values():
            result = (result + val) % MOD
        
        return result