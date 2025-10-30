class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        from string import ascii_lowercase
        
        n = len(s)
        def compute_partitions(s):
            count = 0
            i = 0
            while i < n:
                chars = set()
                j = i
                while j < n:
                    if s[j] not in chars and len(chars) == k:
                        break
                    chars.add(s[j])
                    j +=1
                count +=1
                i = j
            return count

        max_partitions = compute_partitions(s)
        s_list = list(s)
        original_char = s_list[:]
        for i in range(n):
            original_c = s_list[i]
            for c in ascii_lowercase:
                if c == original_c:
                    continue
                s_list[i] = c
                partitions = compute_partitions(s_list)
                if partitions > max_partitions:
                    max_partitions = partitions
            s_list[i] = original_c  # restore original character
        return max_partitions