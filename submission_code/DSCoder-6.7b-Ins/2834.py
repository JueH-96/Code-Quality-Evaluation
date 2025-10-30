class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Create a dictionary to store the count of marbles at each position
        marbles = {}
        for num in nums:
            marbles[num] = marbles.get(num, 0) + 1

        # Apply the moves
        for f, t in zip(moveFrom, moveTo):
            # Decrement the count of marbles at the source position
            marbles[f] -= 1
            # If there are no marbles at the source position, remove it from the dictionary
            if marbles[f] == 0:
                del marbles[f]
            # Increment the count of marbles at the destination position
            marbles[t] = marbles.get(t, 0) + 1

        # Return the sorted list of occupied positions
        return sorted(marbles.keys())