import sys

def main():
    """
    Solves the problem by sequentially calculating the defeat time for each enemy
    using a cumulative damage formula and binary search.
    """
    try:
        # Read input from standard input.
        # sys.stdin.readline is often faster for large inputs.
        N_str = sys.stdin.readline()
        if not N_str: return
        N = int(N_str)
        
        H_str = sys.stdin.readline()
        if not H_str: return
        H = list(map(int, H_str.split()))

    except (IOError, ValueError):
        # Gracefully handle empty input.
        return

    # T_current stores the time when the previous enemy was defeated.
    # It's initialized to 0, as no time has passed before fighting the first enemy.
    T_current = 0

    # The cumulative damage D(t) after t turns is t + 2 * (t // 3).
    # We iterate through each enemy to find its defeat time.
    for i in range(N):
        health = H[i]
        
        # Calculate the total damage dealt up to the point we start attacking enemy i.
        # This is the damage dealt up to time T_current.
        D_start = T_current + 2 * (T_current // 3)
        
        # To defeat enemy i, the total cumulative damage must be at least
        # the damage already dealt plus the enemy's health.
        TargetDamage = health + D_start
        
        # We need to find the smallest time T_end > T_current such that D(T_end) >= TargetDamage.
        # Since D(t) is monotonic, we can use binary search.
        
        # Set the search range for T_end.
        # Lower bound: at least one turn is needed.
        low = T_current + 1
        
        # Upper bound: Each turn deals at least 1 damage, so it takes at most `health` turns.
        # Thus, a safe upper bound for T_end is T_current + health.
        high = T_current + health
        
        # ans_T_end will store the result of the binary search.
        ans_T_end = high 
        
        while low <= high:
            # Use `low + (high - low) // 2` to prevent potential overflow in other languages.
            # It's a good practice, though not strictly necessary in Python.
            mid = low + (high - low) // 2
            
            # Calculate cumulative damage at the time `mid`.
            current_damage_at_mid = mid + 2 * (mid // 3)
            
            if current_damage_at_mid >= TargetDamage:
                # `mid` is a potential answer. We try for a smaller time.
                ans_T_end = mid
                high = mid - 1
            else:
                # `mid` is not enough time. We need more time.
                low = mid + 1
        
        # `ans_T_end` is the minimum time to defeat enemy i. Update T_current for the next enemy.
        T_current = ans_T_end

    # The final T_current after defeating the last enemy is the answer.
    print(T_current)

if __name__ == "__main__":
    main()