import heapq
from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        num_chars = 26  # Number of characters in the alphabet 'a' through 'z'
        
        # min_costs[i][j] will store the minimum cost to transform character with alphabet index i
        # to character with alphabet index j.
        # Initialize all costs to infinity.
        min_costs = [[float('inf')] * num_chars for _ in range(num_chars)]

        # Precompute all-pairs shortest paths using Dijkstra's algorithm
        # We run Dijkstra from each possible starting character (0 to 25)
        for start_char_idx in range(num_chars):
            # For the current start_char_idx, this array will hold the shortest distances
            # from start_char_idx to all other characters.
            # We initialize the distance from start_char_idx to itself as 0.
            min_costs[start_char_idx][start_char_idx] = 0
            
            # Priority queue stores tuples of (cost, char_index)
            # The smallest cost is always at the top.
            pq = [(0, start_char_idx)]

            while pq:
                current_cost, u_idx = heapq.heappop(pq)

                # If we have already found a shorter path to u_idx, skip this one
                if current_cost > min_costs[start_char_idx][u_idx]:
                    continue

                # Explore shifting to the next letter
                # The next character wraps around from 'z' to 'a'
                next_char_idx = (u_idx + 1) % num_chars
                cost_to_next = nextCost[u_idx] # Cost is based on the current character u_idx

                # If a shorter path to next_char_idx is found
                if min_costs[start_char_idx][u_idx] + cost_to_next < min_costs[start_char_idx][next_char_idx]:
                    min_costs[start_char_idx][next_char_idx] = min_costs[start_char_idx][u_idx] + cost_to_next
                    heapq.heappush(pq, (min_costs[start_char_idx][next_char_idx], next_char_idx))

                # Explore shifting to the previous letter
                # The previous character wraps around from 'a' to 'z'
                prev_char_idx = (u_idx - 1 + num_chars) % num_chars # Add num_chars to handle negative results before modulo
                cost_to_prev = previousCost[u_idx] # Cost is based on the current character u_idx

                # If a shorter path to prev_char_idx is found
                if min_costs[start_char_idx][u_idx] + cost_to_prev < min_costs[start_char_idx][prev_char_idx]:
                    min_costs[start_char_idx][prev_char_idx] = min_costs[start_char_idx][u_idx] + cost_to_prev
                    heapq.heappush(pq, (min_costs[start_char_idx][prev_char_idx], prev_char_idx))
        
        # After filling the min_costs table, calculate the total shift distance
        total_shift_distance = 0
        for i in range(len(s)):
            # Convert characters to their 0-indexed alphabet positions
            s_char_idx = ord(s[i]) - ord('a')
            t_char_idx = ord(t[i]) - ord('a')
            
            # Add the precomputed minimum cost for this character transformation
            total_shift_distance += min_costs[s_char_idx][t_char_idx]
            
        return total_shift_distance