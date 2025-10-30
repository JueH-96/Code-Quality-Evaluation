class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0          # total distance travelled in km
        liters_used = 0       # total liters already consumed from the main tank
        
        while mainTank > 0:               # as long as there is fuel in the main tank
            mainTank -= 1                 # consume 1 liter
            distance += 10                # 10 km per liter
            liters_used += 1              # update total consumption
            
            # every 5 liters consumed we try to inject 1 liter from additional tank
            if liters_used % 5 == 0 and additionalTank > 0:
                mainTank += 1             # transfer 1 liter to main tank
                additionalTank -= 1       # decrease additional tank
        
        return distance