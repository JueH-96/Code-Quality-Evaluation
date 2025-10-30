from typing import List
import heapq

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Calculate the distance from knight to each pawn
        distances = []
        for x, y in positions:
            distance = max(abs(kx - x), abs(ky - y))
            heapq.heappush(distances, -distance)

        # Alice and Bob take turns to capture the pawn
        total_moves = 0
        while distances:
            # Alice's turn
            heapq.heappop(distances)
            total_moves += 1

            # Bob's turn
            if distances:
                heapq.heappop(distances)
                total_moves += 1

        return total_moves