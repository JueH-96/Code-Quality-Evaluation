class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        if not groups:
            return []
        n = len(groups)
        max_val = max(groups)
        min_index_d = [10**9] * (max_val + 1)
        
        for j, e in enumerate(elements):
            if e <= max_val and j < min_index_d[e]:
                min_index_d[e] = j
        
        best = [10**9] * (max_val + 1)
        
        for d in range(1, max_val + 1):
            current_min = min_index_d[d]
            if current_min == 10**9:
                continue
            for g in range(d, max_val + 1, d):
                if current_min < best[g]:
                    best[g] = current_min
        
        ans = []
        for g_val in groups:
            if best[g_val] == 10**9:
                ans.append(-1)
            else:
                ans.append(best[g_val])
        return ans