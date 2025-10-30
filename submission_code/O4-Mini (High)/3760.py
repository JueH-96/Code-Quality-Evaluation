class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        INF = 10**9
        n = len(groups)
        # Quick edge‐case checks (though constraints guarantee at least one group and one element)
        if n == 0:
            return []
        if not elements:
            return [-1] * n

        max_group = max(groups)
        max_element = max(elements)

        # Build an array min_idx so that min_idx[v] = smallest index j with elements[j] == v,
        # or INF if v does not appear in elements.
        min_idx = [INF] * (max_element + 1)
        for j, v in enumerate(elements):
            if j < min_idx[v]:
                min_idx[v] = j

        # Map each group‐size g to the list of indices i where groups[i] == g.
        groups_by_size = [[] for _ in range(max_group + 1)]
        for i, g in enumerate(groups):
            groups_by_size[g].append(i)

        # assigned_idx[i] will track the best (smallest) element‐index found for group i.
        assigned_idx = [INF] * n

        # For each element value d that actually appears, propagate its index j = min_idx[d]
        # to all groups whose size is a multiple of d.
        for d in range(1, max_element + 1):
            j = min_idx[d]
            if j == INF:
                continue
            # Walk through multiples of d up to max_group
            for g in range(d, max_group + 1, d):
                for i in groups_by_size[g]:
                    if assigned_idx[i] > j:
                        assigned_idx[i] = j

        # Build the final result, translating INF → -1
        result = []
        for x in assigned_idx:
            result.append(x if x != INF else -1)
        return result