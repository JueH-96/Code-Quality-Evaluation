class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def partition_count(s_val, k_val):
            if not s_val or k_val == 0:
                return 0
            count = 0
            start = 0
            n = len(s_val)
            while start < n:
                freq = [0] * 26
                distinct = 0
                end = start
                while end < n:
                    c = s_val[end]
                    idx = ord(c) - ord('a')
                    if freq[idx] == 0:
                        distinct += 1
                    freq[idx] += 1
                    if distinct > k_val:
                        break
                    end += 1
                count += 1
                start = end
            return count
        
        original = partition_count(s, k)
        max_partitions = original
        
        # Optimization: track if we already found a better partition and can early-exit
        # However, given the problem constraints, we proceed fully.
        n = len(s)
        for i in range(n):
            original_char = s[i]
            # Try all possible lowercase letters except the current one
            for c_code in range(26):
                new_char = chr(ord('a') + c_code)
                if new_char == original_char:
                    continue
                # Build the new string
                # Efficient string manipulation using list
                s_list = list(s)
                s_list[i] = new_char
                new_s = ''.join(s_list)
                current = partition_count(new_s, k)
                if current > max_partitions:
                    max_partitions = current
                    # Early exit if we can't get better than current max_partitions
                # Optimization: break if max_partitions can't be improved
        return max_partitions