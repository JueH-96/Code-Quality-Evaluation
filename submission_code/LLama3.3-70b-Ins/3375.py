from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # Initialize a set to store unique amounts
        amounts = set()
        
        # Iterate over each coin denomination
        for coin in coins:
            # Initialize a queue with the coin denomination
            queue = [coin]
            # Initialize a set to store visited amounts for the current coin
            visited = {coin}
            
            # Perform BFS to generate amounts using the current coin
            while queue:
                # Dequeue the current amount
                amount = queue.pop(0)
                # Add the current amount to the set of all amounts
                amounts.add(amount)
                
                # If the size of the set of all amounts is equal to k, return the kth smallest amount
                if len(amounts) == k:
                    return sorted(amounts)[k-1]
                
                # Generate the next amount by adding the coin denomination to the current amount
                next_amount = amount + coin
                # If the next amount has not been visited, mark it as visited and enqueue it
                if next_amount not in visited:
                    visited.add(next_amount)
                    queue.append(next_amount)
        
        # If k is larger than the number of unique amounts, return -1
        return -1