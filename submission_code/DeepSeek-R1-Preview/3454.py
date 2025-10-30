class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        delta = [t - n for t, n in zip(target, nums)]
        total = 0
        current_block_sign = None
        current_block_max = 0
        
        for d in delta:
            if d == 0:
                if current_block_sign is not None:
                    total += current_block_max
                    current_block_sign = None
                    current_block_max = 0
            else:
                sign = 1 if d > 0 else -1
                if sign != current_block_sign:
                    if current_block_sign is not None:
                        total += current_block_max
                    current_block_sign = sign
                    current_block_max = abs(d)
                else:
                    current_block_max = max(current_block_max, abs(d))
        
        if current_block_sign is not None:
            total += current_block_max
        
        return total