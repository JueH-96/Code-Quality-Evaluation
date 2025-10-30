class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        fuel_consumed_main = 0
        while mainTank > 0:
            mainTank -= 1
            distance += 10
            fuel_consumed_main += 1

            if fuel_consumed_main % 5 == 0 and additionalTank > 0:
                mainTank += 1
                additionalTank -= 1
        return distance