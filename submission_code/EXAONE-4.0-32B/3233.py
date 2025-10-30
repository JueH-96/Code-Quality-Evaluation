def partition_with_change(s, k, change_idx, change_char):
    n = len(s)
    i = 0
    partitions = 0
    while i < n:
        if partitions + (n - i) <= getattr(inspect, 'best', 0):
            break
        partitions += 1
        distinct = 0
        freq = [0] * 26
        j = i
        while j < n:
            if j == change_idx:
                char = change_char
            else:
                char = s[j]
            idx_char = ord(char) - ord('a')
            if freq[idx_char] == 0:
                if distinct < k:
                    distinct += 1
                    freq[idx_char] = 1
                else:
                    break
            else:
                freq[idx_char] += 1
            j += 1
        i = j
    return partitions

class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        best = 0
        global inspect
        import inspect
        part0 = partition_with_change(s, k, -1, None)
        best = max(best, part0)
        inspect.best = best
        
        for idx in range(n):
            for c in "abcdefghijklmnopqrstuvwxyz":
                if c == s[idx]:
                    continue
                part_val = partition_with_change(s, k, idx, c)
                if part_val > best:
                    best = part_val
                    inspect.best = best
        return best