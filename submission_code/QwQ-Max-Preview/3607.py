class Solution:
    def minOperations(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        max_spf = 10**6 + 1
        spf = [0] * max_spf
        for i in range(2, max_spf):
            if spf[i] == 0:
                spf[i] = i
                for j in range(i * i, max_spf, i):
                    if spf[j] == 0:
                        spf[j] = i
        
        total_steps = 0
        prev = None
        
        for x in reversed(nums):
            if x == 1:
                if prev is None:
                    prev = x
                else:
                    if 1 > prev:
                        return -1
                    else:
                        prev = x
                continue
            
            if prev is None:
                prev = x
                continue
            
            if x <= prev:
                prev = x
                continue
            
            steps, new_x = self.compute_steps(x, prev, spf)
            if steps == -1:
                return -1
            total_steps += steps
            prev = new_x
        
        return total_steps
    
    def compute_steps(self, x, T, spf):
        steps = 0
        current = x
        
        while True:
            if current <= T:
                return (steps, current)
            if current == 1:
                return (-1, 0)
            
            spf_current = spf[current]
            next_val = spf_current
            
            if next_val == current:
                return (-1, 0)
            
            steps += 1
            current = next_val