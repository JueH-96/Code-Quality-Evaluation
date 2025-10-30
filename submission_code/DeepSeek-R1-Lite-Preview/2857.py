class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        consumed = 0
        while mainTank >= 5 and additionalTank >= 1:
            consumed += 5
            mainTank -= 5
            mainTank += 1
            additionalTank -= 1
        consumed += mainTank
        total_distance = consumed * 10
        return total_distance