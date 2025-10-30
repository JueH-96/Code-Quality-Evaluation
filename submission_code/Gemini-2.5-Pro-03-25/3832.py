import heapq
import math
from typing import List

class Solution:
    """
    Finds the minimum cost to reach each position i in the line.

    Args:
      cost: A list of integers where cost[i] is the cost to swap with person i.

    Returns:
      A list of integers where answer[i] is the minimum total cost to reach
      position i.
    """
    def minCosts(self, cost: List[int]) -> List[int]:
        """
        Calculates the minimum cost to reach each position using Dijkstra's algorithm.

        The problem can be modeled as finding the shortest paths from a starting node 'n'
        (our initial position) in a graph representing positions and swap costs.

        Graph Representation:
        - Nodes: 0, 1, ..., n, representing the possible positions we can occupy.
                 Initially, we are at position n. Persons 0 to n-1 are at positions 0 to n-1.
        - Edges: Represent possible swaps from our current position 'p'.
            - Edge from 'p' to 'i' (where i < p):
              Represents swapping with person 'i' who is in front of us (originally at position i).
              We pay cost[i]. Our new position becomes 'i'.
              The weight of this edge is cost[i].
            - Edge from 'p' to 'j' (where j > p):
              Represents person 'j' who is behind us (originally at position j) swapping with us for free.
              They initiate the swap. Our new position becomes 'j'.
              The weight of this edge is 0.

        Algorithm:
        - We use Dijkstra's algorithm because it finds the shortest paths from a single
          source node (node 'n') to all other nodes in a graph with non-negative edge weights.
          The swap costs (cost[i] >= 1) and free swaps (cost 0) ensure non-negative weights.
        """
        n = len(cost)
        
        # dist[k] stores the minimum cost found so far to reach position k.
        # Initialize all distances to infinity, signifying they are unreachable initially.
        dist = [math.inf] * (n + 1)
        
        # The cost to reach the starting position 'n' is 0.
        dist[n] = 0
        
        # Priority queue (min-heap) stores tuples of (current_cost, current_position).
        # It allows efficient retrieval of the node with the minimum cost found so far.
        # Start with the initial state: cost 0 at position n.
        pq = [(0, n)] 
        
        while pq:
            # Extract the node 'u' (position) with the smallest distance 'd' (cost)
            # from the priority queue.
            d, u = heapq.heappop(pq)
            
            # If the extracted distance 'd' is greater than the already known shortest
            # distance to 'u' (dist[u]), it means we have found a shorter path to 'u'
            # previously. We can ignore this older, longer path entry.
            if d > dist[u]:
                continue
                
            # Explore potential moves (swaps) from the current position 'u'.
            # This involves relaxing the edges outgoing from node 'u'.

            # Case 1: Move forward by swapping with person 'v' (where v < u).
            # 'v' represents the index of the person (and their initial position).
            # We initiate this swap, pay cost[v], and move to position 'v'.
            for v in range(u):
                edge_cost = cost[v]
                new_dist = dist[u] + edge_cost # Cost to reach v via u
                
                # Relaxation step: If this path through 'u' to 'v' is shorter than
                # the current known shortest path to 'v', update dist[v].
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    # Add the updated cost and position 'v' to the priority queue
                    # for further exploration.
                    heapq.heappush(pq, (new_dist, v))
                    
            # Case 2: Move backward by swapping with person 'v' (where v > u).
            # Person 'v' (behind us) initiates a free swap. We move to position 'v'.
            # 'v' here represents the target position we move to (which is also the
            # initial position of the person we swap with).
            # Note: The positions range from 0 to n.
            for v in range(u + 1, n + 1): 
                edge_cost = 0 # The swap is free for us.
                new_dist = dist[u] + edge_cost # Cost to reach v via u is dist[u].
                
                # Relaxation step: If this path through 'u' to 'v' is shorter, update dist[v].
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    # Add the updated cost and position 'v' to the priority queue.
                    heapq.heappush(pq, (new_dist, v))

        # The problem asks for the minimum costs to reach positions 0 to n-1.
        # The 'dist' array contains costs for positions 0 to n.
        # Return the slice corresponding to positions 0 to n-1.
        return dist[:n]