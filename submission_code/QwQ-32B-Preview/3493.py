class Solution:
    def maxOperations(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        blocks = []
        current_start = -1
        for i in range(n):
            if s[i] == '1':
                if current_start == -1:
                    current_start = i
            else:
                if current_start != -1:
                    blocks.append((current_start, i - 1))
                    current_start = -1
        if current_start != -1:
            blocks.append((current_start, n - 1))
        
        total_operations = 0
        for i in range(len(blocks) - 1):
            left_block = blocks[i]
            right_block = blocks[i + 1]
            zeros_between = right_block[0] - left_block[1] - 1
            ones_in_left = left_block[1] - left_block[0] + 1
            total_operations += ones_in_left * zeros_between
        return total_operations