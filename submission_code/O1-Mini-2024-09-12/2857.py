class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        mileage = 10
        while mainTank >= 5:
            distance += 5 * mileage
            mainTank -= 5
            if additionalTank >= 1:
                mainTank += 1
                additionalTank -= 1
        distance += mainTank * mileage
        return distance