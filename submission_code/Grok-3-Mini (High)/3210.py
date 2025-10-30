import collections

class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(s)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            if s[i] in vowels:
                prefix_sum[i + 1] = prefix_sum[i] + 1
            else:
                prefix_sum[i + 1] = prefix_sum[i] - 1
        
        sum_to_indices = collections.defaultdict(list)
        for i in range(n + 1):
            sum_to_indices[prefix_sum[i]].append(i)
        
        count = 0
        for idx_list in sum_to_indices.values():
            m = len(idx_list)
            for i in range(m):
                for j in range(i + 1, m):
                    dist = idx_list[j] - idx_list[i]
                    v = dist // 2
                    if (v * v) % k == 0:
                        count += 1
        
        return count