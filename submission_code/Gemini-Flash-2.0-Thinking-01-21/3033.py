class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diff = []
        ones_count = 0
        for i in range(n):
            if s1[i] != s2[i]:
                diff.append('1')
                ones_count += 1
            else:
                diff.append('0')
        if ones_count % 2 != 0:
            return -1
        diff_str = "".join(diff)
        blocks = []
        i = 0
        while i < n:
            if diff_str[i] == '1':
                start_index = i
                while i < n and diff_str[i] == '1':
                    i += 1
                blocks.append(i - start_index)
            else:
                i += 1
        cost_op2 = 0
        odd_blocks_count = 0
        for length in blocks:
            cost_op2 += length // 2
            if length % 2 == 1:
                odd_blocks_count += 1
        cost_op1 = (odd_blocks_count // 2) * x
        return cost_op2 + cost_op1