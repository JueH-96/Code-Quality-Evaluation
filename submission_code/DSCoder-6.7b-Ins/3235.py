class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        if len(source) != len(target):
            return -1
        cost_dict = {original[i]: min(cost[i], changed[i]) for i in range(len(original))}
        min_cost = 0
        for s, t in zip(source, target):
            if s == t:
                continue
            if s not in cost_dict or t not in cost_dict or cost_dict[s] > cost_dict[t]:
                return -1
            min_cost += cost_dict[s]
        return min_cost