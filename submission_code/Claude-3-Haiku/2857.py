class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        while mainTank > 0:
            distance += min(mainTank, 5) * 10
            mainTank -= min(mainTank, 5)
            if mainTank == 0 and additionalTank > 0:
                mainTank += 1
                additionalTank -= 1
        return distance