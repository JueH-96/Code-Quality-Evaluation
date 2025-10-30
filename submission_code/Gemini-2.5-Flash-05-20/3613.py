import collections
from typing import List, Dict, Set, Tuple

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        
        # 1. Collect all unique currencies to manage their amounts
        all_currencies: Set[str] = set()
        all_currencies.add(initialCurrency)
        
        for pair_list in [pairs1, pairs2]:
            for pair in pair_list:
                all_currencies.add(pair[0])
                all_currencies.add(pair[1])
            
        # 2. Build graphs for Day 1 and Day 2, including inverse conversions
        # graph[currency] = [(target_currency, rate), ...]
        graph1: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
        for i in range(len(pairs1)):
            c1, c2 = pairs1[i]
            rate = rates1[i]
            graph1[c1].append((c2, rate))
            graph1[c2].append((c1, 1.0 / rate)) # Inverse conversion
            
        graph2: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
        for i in range(len(pairs2)):
            c1, c2 = pairs2[i]
            rate = rates2[i]
            graph2[c1].append((c2, rate))
            graph2[c2].append((c1, 1.0 / rate)) # Inverse conversion
            
        # 3. Phase 1: Day 1 Conversions (SPFA Algorithm)
        max_amounts_day1: Dict[str, float] = {currency: 0.0 for currency in all_currencies}
        max_amounts_day1[initialCurrency] = 1.0 # Start with 1.0 of initialCurrency
        
        # Queue for SPFA and a set to track currencies currently in the queue
        queue: collections.deque[str] = collections.deque([initialCurrency])
        in_queue: Set[str] = {initialCurrency}
        
        while queue:
            u = queue.popleft()
            in_queue.discard(u) # Remove from in_queue set
            
            for v, rate in graph1[u]:
                new_amount = max_amounts_day1[u] * rate
                if new_amount > max_amounts_day1[v]:
                    max_amounts_day1[v] = new_amount
                    if v not in in_queue:
                        queue.append(v)
                        in_queue.add(v)
                        
        # 4. Phase 2: Day 2 Conversions (SPFA Algorithm)
        # Initialize Day 2 amounts with the maximums from Day 1
        max_amounts_day2: Dict[str, float] = {currency: 0.0 for currency in all_currencies}
        queue = collections.deque()
        in_queue = set()

        # Add currencies that have a non-zero amount after Day 1 to the Day 2 starting queue
        for currency, amount in max_amounts_day1.items():
            if amount > 0: # Only propagate if we have a positive amount
                max_amounts_day2[currency] = amount
                queue.append(currency)
                in_queue.add(currency)
        
        while queue:
            u = queue.popleft()
            in_queue.discard(u)
            
            for v, rate in graph2[u]:
                new_amount = max_amounts_day2[u] * rate
                if new_amount > max_amounts_day2[v]:
                    max_amounts_day2[v] = new_amount
                    if v not in in_queue:
                        queue.append(v)
                        in_queue.add(v)
                        
        # 5. Return the maximum amount of initialCurrency after all conversions
        return max_amounts_day2[initialCurrency]