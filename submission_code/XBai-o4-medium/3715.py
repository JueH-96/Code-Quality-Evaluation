from typing import List

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Sort the coins by their starting positions
        coins.sort()
        
        # Generate the blocks including gaps with 0 coins
        blocks = []
        prev_end = 0
        for l, r, c in coins:
            if l > prev_end + 1:
                blocks.append((prev_end + 1, l - 1, 0))
            blocks.append((l, r, c))
            prev_end = r
        
        # Compute the prefix sum array
        prefix_sum = [0] * (len(blocks) + 1)
        for i in range(len(blocks)):
            l, r, c = blocks[i]
            prefix_sum[i + 1] = prefix_sum[i] + (r - l + 1) * c
        
        # Generate candidate starting positions
        candidates = set()
        for l, r, _ in blocks:
            candidates.add(l)
            s_candidate = r - k + 1
            candidates.add(s_candidate)
        
        max_total = 0
        
        # Evaluate each candidate
        for s in candidates:
            e = s + k - 1
            block_start = self.find_block(s, blocks)
            block_end = self.find_block(e, blocks)
            
            if block_start == block_end:
                if block_start == -1 or block_start == len(blocks):
                    current = 0
                else:
                    bl, br, bc = blocks[block_start]
                    current = (e - s + 1) * bc
                if current > max_total:
                    max_total = current
            else:
                sum_start = 0
                if block_start != -1:
                    bl, br, bc = blocks[block_start]
                    sum_start = (br - s + 1) * bc
                
                sum_end = 0
                if block_end != len(blocks):
                    bl, br, bc = blocks[block_end]
                    sum_end = (e - bl + 1) * bc
                
                sum_middle = 0
                if block_start + 1 <= block_end - 1:
                    sum_middle = prefix_sum[block_end] - prefix_sum[block_start + 1]
                
                total = sum_start + sum_end + sum_middle
                if total > max_total:
                    max_total = total
        
        return max_total
    
    def find_block(self, x, blocks):
        left = 0
        right = len(blocks) - 1
        while left <= right:
            mid = (left + right) // 2
            bl, br, _ = blocks[mid]
            if x < bl:
                right = mid - 1
            elif x > br:
                left = mid + 1
            else:
                return mid
        # After loop, check if x is before all blocks or after
        if left == 0:
            return -1
        else:
            return len(blocks)