class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        power.sort()
        n = len(power)
        used = [False] * n
        total_damage = 0
        
        for i in range(n):
            if used[i]:
                continue
            
            max_damage_for_current = 0
            best_spell_index = -1

            
            current_damage = power[i]
            max_damage_for_current = current_damage
            best_spell_index = i
            used[i] = True

            
            for j in range(i + 1, n):
                if not used[j] and abs(power[j] - power[i]) >= 3:
                    if power[j] + current_damage > max_damage_for_current:
                        max_damage_for_current = power[j] + current_damage
                        best_spell_index = j

            if best_spell_index != -1:
                total_damage += max_damage_for_current
                used[best_spell_index] = True

        return total_damage