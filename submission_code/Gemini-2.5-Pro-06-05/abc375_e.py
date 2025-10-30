# YOUR CODE HERE
import sys

def solve():
    """
    Solves the team strength balancing problem using dynamic programming.
    """
    try:
        # Fast I/O
        readline = sys.stdin.readline
        
        # Read N
        n_str = readline()
        if not n_str:
            return
        N = int(n_str)
        
        # Read people's team and strength
        people = []
        total_strength = 0
        for _ in range(N):
            a, b = map(int, readline().split())
            people.append((a, b))
            total_strength += b
            
    except (IOError, ValueError):
        # Handle potential empty input or parsing errors
        return

    # The total strength must be divisible by 3 for three equal teams.
    if total_strength % 3 != 0:
        print(-1)
        return

    target_strength = total_strength // 3
    T = target_strength
    
    # A value for infinity, larger than any possible number of switches (which is at most N).
    inf = N + 1

    # dp[j][k] = minimum number of switches to get team 1 strength j and team 2 strength k.
    # The strength of team 3 is implicitly determined by the total strength.
    dp = [[inf] * (T + 1) for _ in range(T + 1)]
    
    # Base case: 0 strength for both teams with 0 people requires 0 switches.
    dp[0][0] = 0

    # Iterate through each person, updating the DP states.
    for p_team, p_strength in people:
        # Create a new DP table for the state after considering the current person.
        # This is necessary to avoid using partially updated information from the current iteration.
        dp_new = [[inf] * (T + 1) for _ in range(T + 1)]
        
        for j in range(T + 1):
            for k in range(T + 1):
                # If the state (j, k) was not reachable before, we can't extend from it.
                if dp[j][k] == inf:
                    continue
                
                cost_before = dp[j][k]
                
                # Consider three possibilities for the current person:
                
                # Option 1: Place the person in team 1
                new_j = j + p_strength
                if new_j <= T:
                    switches = 1 if p_team != 1 else 0
                    dp_new[new_j][k] = min(dp_new[new_j][k], cost_before + switches)
                
                # Option 2: Place the person in team 2
                new_k = k + p_strength
                if new_k <= T:
                    switches = 1 if p_team != 2 else 0
                    dp_new[j][new_k] = min(dp_new[j][new_k], cost_before + switches)
                    
                # Option 3: Place the person in team 3
                # The strengths (j, k) of teams 1 and 2 don't change for this option.
                switches = 1 if p_team != 3 else 0
                dp_new[j][k] = min(dp_new[j][k], cost_before + switches)
        
        # The new DP table becomes the current one for the next iteration.
        dp = dp_new
        
    # The final answer is the number of switches for the state where
    # team 1 and team 2 both have the target strength.
    result = dp[T][T]

    if result == inf:
        print(-1)
    else:
        print(result)

solve()