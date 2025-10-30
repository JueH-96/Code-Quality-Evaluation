class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        if k == 0:
            return 0
        
        def compute_jumps(s):
            jumps = [0] * n
            count = {}
            left = 0
            distinct = 0
            for right in range(n):
                c = s[right]
                if c not in count or count[c] == 0:
                    distinct += 1
                count[c] = count.get(c, 0) + 1
                while distinct > k:
                    left_c = s[left]
                    count[left_c] -= 1
                    if count[left_c] == 0:
                        distinct -= 1
                    left += 1
                jumps[right] = left
            return jumps
        
        def count_partitions(jumps):
            partitions = 0
            pos = 0
            while pos < n:
                partitions += 1
                if pos >= n:
                    break
                leftmost = jumps[pos]
                max_jump = pos
                for i in range(pos, n):
                    if jumps[i] <= pos:
                        max_jump = i
                pos = max_jump + 1
            return partitions
        
        original_jumps = compute_jumps(s)
        original_partitions = count_partitions(original_jumps)
        max_partitions = original_partitions
        
        for i in range(n):
            original_char = s[i]
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c == original_char:
                    continue
                new_s = s[:i] + c + s[i+1:]
                new_jumps = compute_jumps(new_s)
                partitions = count_partitions(new_jumps)
                if partitions > max_partitions:
                    max_partitions = partitions
        
        return max_partitions