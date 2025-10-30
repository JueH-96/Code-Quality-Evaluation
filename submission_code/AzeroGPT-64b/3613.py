from collections import defaultdict
from heapq import heappop, heappush

def dijkstra(pairs, rates, source, target, n):
    graph = defaultdict(list)
    for (u, v), rate in zip(pairs, rates):
        graph[u].append((v, rate))
    
    visited = [False] * n
    max_rates = [-float('inf')] * n
    max_rates[source] = 0
    priority_queue = [(-0, source)]
    
    while priority_queue:
        max_rate, current = heappop(priority_queue)
        if not visited[current]:
            visited[current] = True
            for neighbor, rate in graph[target_codes[current]]:
                index = target_codes[neighbor]
                new_rate = max_rate + math.log(rate)
                if new_rate > max_rates[index]:
                    max_rates[index] = new_rate
                    heappush(priority_queue, (new_rate, index))
    
    return math.exp(max_rates[target]) if visited[target] else -1

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        global target_codes
        source = None
        codes = []
        codes.append(initialCurrency)
        target_codes = {}
        
        target_codes[initialCurrency] = 0
        
        n1 = len(pairs1)
        n2 = len(pairs2)
        
        rates = []
        pairs = []
        pairs.extend(pairs1)
        pairs.extend(pairs2)
        rates.extend(rates1)
        rates.extend(rates2)
        
        for i in range(n1 + n2):
            if pairs[i][0] not in target_codes:
                codes.append(pairs[i][0])
                target_codes[pairs[i][0]] = len(codes) - 1
            if pairs[i][1] not in target_codes:
                codes.append(pairs[i][1])
                target_codes[pairs[i][1]] = len(codes) - 1
                
        if source != target_codes[initialCurrency]:
            source = target_codes[initialCurrency]
            
        n = len(codes)
        max_rate1 = dijkstra(pairs1, rates1, source, target_codes[initialCurrency], n)
        max_rate2 = dijkstra(pairs2, rates2, source, target_codes[initialCurrency], n)
        
        return max(1.0, max_rate1, max_rate2)