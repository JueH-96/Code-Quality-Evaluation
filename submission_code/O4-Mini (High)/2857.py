class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        # Each time we burn 5 liters from the main tank and if additionalTank > 0,
        # we transfer 1 liter back into the main tank.  Net loss per such "injection"
        # is 4 liters from the combined supply.  We can only do as many injections
        # as additionalTank allows, and as long as the main tank stays >=5 before each injection.
        # It can be shown that the maximum injections = min(additionalTank, (mainTank-1)//4).
        injections = min(additionalTank, (mainTank - 1) // 4)
        # Total liters burned = initial mainTank + injections.  Each liter gives 10 km.
        return (mainTank + injections) * 10