class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Count occurrences of each type of move
        left_moves  = moves.count('L')
        right_moves = moves.count('R')
        free_moves  = moves.count('_')   # moves we can assign either direction

        # Fixed displacement contributed by mandatory moves
        fixed_displacement = right_moves - left_moves   # right is +1, left is -1

        # To maximize the absolute distance, assign all free moves
        # in the direction of the current displacement.
        # Resulting maximal distance = |fixed_displacement| + free_moves
        return abs(fixed_displacement) + free_moves