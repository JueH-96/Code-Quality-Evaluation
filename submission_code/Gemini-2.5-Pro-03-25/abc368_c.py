# YOUR CODE HERE
import sys

# Function to compute S(t)
# S(t) is the total damage dealt from time 1 up to time t.
def S(t):
    """
    Calculates the total damage dealt from time 1 to t.
    The damage D(k) at time k is 3 if k is a multiple of 3, and 1 otherwise.
    S(t) = sum_{k=1}^t D(k).
    """
    # If time t is 0 or less, no attacks have happened, so total damage is 0.
    if t <= 0:
        return 0
    
    # Calculate the number of full cycles of 3 attacks completed by time t.
    # A cycle consists of attacks at times 3k-2, 3k-1, 3k (for k >= 1).
    # Example cycle: times 1, 2, 3. Damages: 1, 1, 3. Total damage per cycle = 5.
    # The total number of full cycles completed by time t is q = floor(t/3).
    q = t // 3
    
    # Calculate the number of remaining attacks in the current (possibly incomplete) cycle.
    # r = t mod 3 gives the position within the cycle (0, 1, or 2 steps after the last multiple of 3).
    # Note: t % 3 = 0 means t is a multiple of 3, completing a cycle.
    # t % 3 = 1 means t = 3q + 1.
    # t % 3 = 2 means t = 3q + 2.
    r = t % 3
    
    # Calculate total damage from the full cycles. Each full cycle deals 5 damage.
    # So, q full cycles deal 5 * q damage.
    damage = 5 * q
    
    # Add damage from the remaining steps in the current partial cycle.
    if r == 1:
        # If t = 3q + 1, there is one extra attack at time 3q+1.
        # Since (3q+1) % 3 = 1, the damage D(3q+1) is 1.
        # Total damage S(3q+1) = S(3q) + D(3q+1) = 5*q + 1.
        damage += 1
    elif r == 2:
        # If t = 3q + 2, there are two extra attacks at times 3q+1 and 3q+2.
        # D(3q+1) = 1 (since (3q+1)%3 = 1).
        # D(3q+2) = 1 (since (3q+2)%3 = 2).
        # Total damage S(3q+2) = S(3q) + D(3q+1) + D(3q+2) = 5*q + 1 + 1 = 5*q + 2.
        damage += 2
    # If r == 0, then t = 3q. This means the time t completes a full cycle.
    # The last attack was at time 3q. Damage D(3q) = 3.
    # The total damage S(3q) = 5*q. No extra damage needs to be added beyond the `damage = 5 * q` calculation.
        
    return damage

def solve():
    # Read input N: number of enemies
    # Using sys.stdin.readline for potentially faster input reading
    N = int(sys.stdin.readline())
    
    # Read input H: list of healths of enemies
    # Using list comprehension for parsing the space-separated integers on the line
    H = [int(x) for x in sys.stdin.readline().split()]

    # Initialize T_prev: represents the time T_{i-1} when the (i-1)-th enemy was defeated.
    # For the first enemy (i=0), T_0 = 0, as the game starts at T=0 and the first attack is at T=1.
    T_prev = 0

    # Iterate through each enemy, indexed from 0 to N-1
    for i in range(N):
        # Calculate S_prev = S(T_prev): total cumulative damage dealt up to time T_prev.
        # This is the total damage dealt before starting to attack the current enemy 'i'.
        S_prev = S(T_prev)
        
        # Calculate the target cumulative damage required by the time the current enemy (enemy i) is defeated.
        # To defeat enemy 'i', the total damage dealt from time 1 up to the defeat time T_i must be at least
        # the sum of its health H[i] and the damage S_prev already dealt to previous enemies.
        Target = H[i] + S_prev
        
        # We need to find the minimum time T_curr (which will be T_i) such that:
        # 1. T_curr >= T_prev + 1 (attacks on enemy i start at time T_prev + 1)
        # 2. S(T_curr) >= Target (total damage up to time T_curr is sufficient)
        
        # We use binary search to efficiently find this minimum T_curr.
        # The function S(t) is monotonically increasing with t, which makes binary search applicable.
        
        # Define the search space [low, high] for T_curr.
        # The lower bound is T_prev + 1, as the first attack on enemy i happens at this time.
        low = T_prev + 1
        
        # The upper bound. We established that T_curr <= T_prev + H[i] is sufficient.
        # Reason: The number of attacks k required for enemy i is T_curr - T_prev.
        # Since the minimum damage per attack is 1, at most H[i] attacks are needed to deal H[i] damage (k <= H[i]).
        # Thus, T_curr = T_prev + k <= T_prev + H[i].
        # Python's arbitrary precision integers handle large values of T_prev + H[i].
        high = T_prev + H[i] 
        
        # Variable 'ans' will store the minimum time found satisfying the condition S(mid) >= Target.
        # Initialize 'ans' with 'high'. We know S(high) >= Target is guaranteed, so 'high' is a valid time (possibly not minimum).
        # The binary search will refine 'ans' to the minimum valid time.
        ans = high 
        
        # Standard binary search implementation
        while low <= high:
            # Calculate midpoint. Using this form `low + (high - low) // 2` avoids potential overflow 
            # for large low/high in languages with fixed-size integers. It's generally good practice.
            mid = low + (high - low) // 2
            
            # Check if the cumulative damage S(mid) reaches the target
            if S(mid) >= Target:
                # If S(mid) is sufficient, then 'mid' is a possible time T_curr.
                # Store this 'mid' as a potential answer ('ans').
                # Since we want the *minimum* time, we try searching in the lower half: [low, mid-1].
                ans = mid
                high = mid - 1
            else:
                # If S(mid) is not sufficient, we need more time (and thus more damage).
                # The minimum time must be greater than 'mid'.
                # We search in the upper half: [mid+1, high].
                low = mid + 1
                
        # After the binary search loop finishes, 'ans' holds the minimum time T_curr (which is T_i)
        # required to defeat the i-th enemy.
        # We update T_prev to this value for the calculation related to the next enemy (i+1).
        T_prev = ans 

    # After iterating through all N enemies, T_prev will hold the final time T_N,
    # when the last enemy (enemy N-1) is defeated.
    # Print the final result T_N.
    print(T_prev)

# Call the solve function to execute the program logic
solve()