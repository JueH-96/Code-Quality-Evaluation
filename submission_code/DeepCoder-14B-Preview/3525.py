from typing import List

class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        if n == 0:
            return 0
        
        current_states = {
            ('A', False): energyDrinkA[0],
            ('B', False): energyDrinkB[0],
        }
        
        for i in range(1, n):
            next_states = {}
            for state in current_states:
                current_drink, can_switch = state
                current_total = current_states[state]
                
                if not can_switch:
                    # Option 1: Continue on the current drink
                    new_total = current_total + (energyDrinkA[i] if current_drink == 'A' else energyDrinkB[i])
                    new_state = (current_drink, False)
                    if new_state in next_states:
                        if new_total > next_states[new_state]:
                            next_states[new_state] = new_total
                    else:
                        next_states[new_state] = new_total
                    
                    # Option 2: Switch to the other drink
                    other_drink = 'B' if current_drink == 'A' else 'A'
                    new_total_switch = current_total + (energyDrinkA[i] if current_drink == 'A' else energyDrinkB[i])
                    new_state_switch = (other_drink, True)
                    if new_state_switch in next_states:
                        if new_total_switch > next_states[new_state_switch]:
                            next_states[new_state_switch] = new_total_switch
                    else:
                        next_states[new_state_switch] = new_total_switch
                else:
                    # can_switch is True: transition to current drink with can_switch=False
                    new_total = current_total
                    new_state = (current_drink, False)
                    if new_state in next_states:
                        if new_total > next_states[new_state]:
                            next_states[new_state] = new_total
                    else:
                        next_states[new_state] = new_total
            current_states = next_states
        
        if not current_states:
            return 0
        return max(current_states.values())