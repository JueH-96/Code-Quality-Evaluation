class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        while mainTank >= 5:
            mainTank -= 5
            distance += 50  # 5 liters * 10 km/l
            if additionalTank >= 1:
                mainTank += 1
                additionalTank -= 1
        distance += mainTank * 10
        return distance