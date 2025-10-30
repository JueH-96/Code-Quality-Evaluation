class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        # Sort the skills in ascending order
        skills.sort()

        # Initialize the count of consecutive wins
        count = 0

        # Iterate over the sorted skills
        for i in range(1, len(skills)):
            # If the skill of the current player is higher than the previous one
            if skills[i] > skills[i - 1]:
                # Increment the count of consecutive wins
                count += 1
            else:
                # Reset the count of consecutive wins
                count = 0

            # If the count of consecutive wins is equal to k
            if count == k:
                # Return the index of the current player
                return i

        # If no player has won k games in a row
        return -1