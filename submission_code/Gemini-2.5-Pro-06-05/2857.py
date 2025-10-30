class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        
        total_distance = 0
        
        # This simulation models the fuel consumption process.
        # The loop continues as long as there is enough fuel in the main tank
        # to consume a 5-liter chunk, which is the trigger for a potential
        # transfer from the additional tank.
        while mainTank >= 5:
            
            # The truck consumes 5 liters from the main tank.
            mainTank -= 5
            
            # The distance covered is 5 liters * 10 km/liter = 50 km.
            total_distance += 50
            
            # After consuming 5 liters, a transfer may occur.
            # If the additional tank has fuel, transfer 1 liter to the main tank.
            if additionalTank > 0:
                additionalTank -= 1
                mainTank += 1
        
        # Once the main tank has less than 5 liters, no more 5-liter chunks can be
        # consumed, so no more transfers will happen.
        # The truck uses up all the remaining fuel in the main tank.
        total_distance += mainTank * 10
        
        return total_distance