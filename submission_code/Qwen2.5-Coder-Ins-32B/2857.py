class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        while mainTank > 0:
            distance += 10
            mainTank -= 1
            if distance % 50 == 0 and additionalTank > 0:
                mainTank += 1
                additionalTank -= 1
        return distance