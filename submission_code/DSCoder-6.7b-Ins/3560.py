class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Define the possible moves of the knight
        moves = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]

        # Initialize the queue with the knight's position
        queue = [(kx, ky)]

        # Initialize the maximum moves as 0
        max_moves = 0

        # While there are still pawns on the board
        while positions:
            # Initialize the next level of positions
            next_level = []

            # For each position in the current level
            for x, y in queue:
                # Try all possible moves
                for dx, dy in moves:
                    # Calculate the new position
                    nx, ny = x + dx, y + dy

                    # If the new position is a valid position and a pawn
                    if 0 <= nx < 50 and 0 <= ny < 50 and [nx, ny] in positions:
                        # Add the new position to the next level
                        next_level.append((nx, ny))
                        # Remove the pawn from the positions
                        positions.remove([nx, ny])

            # Update the queue with the next level of positions
            queue = next_level

            # If there are still positions in the queue, increment the maximum moves
            if queue:
                max_moves += 1

        # Return the maximum moves
        return max_moves