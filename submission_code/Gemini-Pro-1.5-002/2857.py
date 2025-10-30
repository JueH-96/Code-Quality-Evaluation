class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        while mainTank > 0:
            spent = min(mainTank, 5)
            distance += spent * 10
            mainTank -= spent
            if spent == 5 and additionalTank > 0:
                mainTank += 1
                additionalTank -= 1
        return distance