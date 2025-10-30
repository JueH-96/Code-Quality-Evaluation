class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 1000000007
        n = len(nums)
        
        def get_min_diff(seq):
            min_diff = float('inf')
            for i in range(len(seq)):
                for j in range(i+1, len(seq)):
                    min_diff = min(min_diff, abs(seq[i] - seq[j]))
            return min_diff
        
        def generate_subsequences(start, curr_seq):
            if len(curr_seq) == k:
                return get_min_diff(curr_seq)
            
            total = 0
            for i in range(start, n):
                curr_seq.append(nums[i])
                total = (total + generate_subsequences(i + 1, curr_seq)) % MOD
                curr_seq.pop()
            return total
        
        return generate_subsequences(0, [])