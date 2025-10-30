import sys

def main():
    N = int(sys.stdin.readline())
    H_values = list(map(int, sys.stdin.readline().split()))

    current_total_T = 0  # This tracks the T value when the previous enemy was defeated.

    for h_enemy in H_values:
        
        t_at_start_of_enemy = current_total_T # T value before starting attacks on this enemy
        
        # Binary search for k_attacks (number of attacks for this enemy)
        # We need the smallest k_attacks such that total damage >= h_enemy.
        
        low_k = 1
        # Max k_attacks can be h_enemy (if all attacks deal 1 damage).
        # Min k_attacks can be ceil(h_enemy / 3) (if all attacks deal 3 damage).
        # The search range [1, h_enemy] is safe.
        high_k = h_enemy  
        
        # Stores the minimum k_attacks found so far that defeats the enemy.
        # Initialize with h_enemy, as it's a guaranteed (though not necessarily minimal) number of attacks.
        min_k_for_this_enemy = h_enemy 

        while low_k <= high_k:
            k_candidate = low_k + (high_k - low_k) // 2
            
            # Calculate damage dealt by k_candidate attacks.
            # The T values for these attacks range from t_at_start_of_enemy + 1 
            # to t_at_start_of_enemy + k_candidate.
            
            # Number of T values that are multiples of 3 in this range:
            # This is equivalent to (count of multiples of 3 up to T_last_attack) - 
            # (count of multiples of 3 up to T_value_before_first_attack).
            # T_last_attack = t_at_start_of_enemy + k_candidate
            # T_value_before_first_attack = t_at_start_of_enemy
            count_strong_attacks = (t_at_start_of_enemy + k_candidate) // 3 - t_at_start_of_enemy // 3
            
            # Total damage = (damage from strong attacks) + (damage from normal attacks)
            # damage = (count_strong_attacks * 3) + ((k_candidate - count_strong_attacks) * 1)
            # damage = count_strong_attacks * 2 + k_candidate
            damage = count_strong_attacks * 2 + k_candidate
            
            if damage >= h_enemy:
                # This k_candidate is sufficient, try to find an even smaller one.
                min_k_for_this_enemy = k_candidate
                high_k = k_candidate - 1 
            else:
                # This k_candidate is not sufficient, need more attacks.
                low_k = k_candidate + 1 
        
        current_total_T += min_k_for_this_enemy

    print(current_total_T)

if __name__ == '__main__':
    main()