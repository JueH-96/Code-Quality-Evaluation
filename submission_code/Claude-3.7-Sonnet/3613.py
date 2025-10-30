class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        from collections import defaultdict
        
        # Build conversion graph for day 1
        graph1 = defaultdict(list)
        for (src, dst), rate in zip(pairs1, rates1):
            graph1[src].append((dst, rate))
            graph1[dst].append((src, 1/rate))  # Add inverse conversion
        
        # Calculate maximum amounts for each currency after day 1
        amounts1 = defaultdict(float)
        amounts1[initialCurrency] = 1.0
        
        # Relaxation for day 1
        for _ in range(20):  # Enough iterations for the given constraints
            updated = False
            for src, edges in graph1.items():
                if amounts1[src] > 0:  # Only consider currencies we have
                    for dst, rate in edges:
                        new_amount = amounts1[src] * rate
                        if new_amount > amounts1[dst]:
                            amounts1[dst] = new_amount
                            updated = True
            if not updated:
                break
        
        # Build conversion graph for day 2
        graph2 = defaultdict(list)
        for (src, dst), rate in zip(pairs2, rates2):
            graph2[src].append((dst, rate))
            graph2[dst].append((src, 1/rate))  # Add inverse conversion
        
        # Start day 2 with amounts from day 1
        amounts2 = amounts1.copy()
        
        # Relaxation for day 2
        for _ in range(20):  # Enough iterations for the given constraints
            updated = False
            for src, edges in graph2.items():
                if amounts2[src] > 0:  # Only consider currencies we have
                    for dst, rate in edges:
                        new_amount = amounts2[src] * rate
                        if new_amount > amounts2[dst]:
                            amounts2[dst] = new_amount
                            updated = True
            if not updated:
                break
        
        return amounts2[initialCurrency]