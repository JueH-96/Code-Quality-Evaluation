class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        from collections import defaultdict
        
        # Build graph for day 1
        graph1 = defaultdict(list)
        for i, (start, end) in enumerate(pairs1):
            graph1[start].append((end, rates1[i]))
            graph1[end].append((start, 1.0 / rates1[i]))
        
        # Find maximum amount of each currency after day 1
        max_after_day1 = {initialCurrency: 1.0}
        changed = True
        while changed:
            changed = False
            for curr in list(max_after_day1.keys()):
                for next_curr, rate in graph1[curr]:
                    new_amount = max_after_day1[curr] * rate
                    if next_curr not in max_after_day1 or new_amount > max_after_day1[next_curr]:
                        max_after_day1[next_curr] = new_amount
                        changed = True
        
        # Build graph for day 2
        graph2 = defaultdict(list)
        for i, (start, end) in enumerate(pairs2):
            graph2[start].append((end, rates2[i]))
            graph2[end].append((start, 1.0 / rates2[i]))
        
        # Start day 2 with the amounts from day 1
        max_day2 = max_after_day1.copy()
        changed = True
        while changed:
            changed = False
            for curr in list(max_day2.keys()):
                for next_curr, rate in graph2[curr]:
                    new_amount = max_day2[curr] * rate
                    if next_curr not in max_day2 or new_amount > max_day2[next_curr]:
                        max_day2[next_curr] = new_amount
                        changed = True
        
        return max_day2.get(initialCurrency, 1.0)