# YOUR CODE HERE
import sys

def solve():
    N = int(sys.stdin.readline())
    
    people_data = []
    total_strength = 0
    for _ in range(N):
        A_i, B_i = map(int, sys.stdin.readline().split())
        people_data.append((A_i, B_i))
        total_strength += B_i
    
    # If total strength is not divisible by 3, it's impossible to balance.
    if total_strength % 3 != 0:
        print(-1)
        return
        
    target_strength = total_strength // 3
    
    # dp[s1][s2] stores the minimum number of people who need to switch teams
    # to achieve strength s1 for Team 1 and s2 for Team 2,
    # considering all people processed so far.
    # The strength for Team 3 is implicitly (current_total_sum_B - s1 - s2).
    
    # Initialize dp table with a value representing infinity.
    # N+1 is a safe upper bound for the number of moves.
    INF = N + 1 
    dp = [[INF for _ in range(target_strength + 1)] for _ in range(target_strength + 1)]
    
    # Base case: 0 strength for both teams initially requires 0 moves.
    # This state represents the assignment of 0 people.
    dp[0][0] = 0
    
    # current_total_sum_B tracks the sum of strengths of people processed so far.
    current_total_sum_B = 0
    
    # Iterate through each person
    for initial_team, strength_B in people_data:
        # Create a new_dp table for the current iteration to store results
        # after considering the current person. This ensures that updates for
        # the current person only use values from the previous state (dp).
        new_dp = [[INF for _ in range(target_strength + 1)] for _ in range(target_strength + 1)]
        
        # Add current person's strength to the running total.
        current_total_sum_B += strength_B
        
        # Iterate over all possible s1 and s2 values from the previous state (dp).
        # We need to consider all states reachable from the previous iteration.
        for s1 in range(target_strength + 1):
            for s2 in range(target_strength + 1):
                # If the previous state (s1, s2) was unreachable, skip it.
                if dp[s1][s2] == INF:
                    continue
                
                # Option 1: Assign current person to Team 1
                # Check if adding current strength_B to s1 exceeds target_strength for Team 1.
                if s1 + strength_B <= target_strength:
                    cost = 1 if initial_team != 1 else 0
                    new_dp[s1 + strength_B][s2] = min(new_dp[s1 + strength_B][s2], dp[s1][s2] + cost)
                
                # Option 2: Assign current person to Team 2
                # Check if adding current strength_B to s2 exceeds target_strength for Team 2.
                if s2 + strength_B <= target_strength:
                    cost = 1 if initial_team != 2 else 0
                    new_dp[s1][s2 + strength_B] = min(new_dp[s1][s2 + strength_B], dp[s1][s2] + cost)
                
                # Option 3: Assign current person to Team 3
                # The strength of Team 3 (from all people processed so far, including current)
                # is (current_total_sum_B - s1 - s2).
                # Check if this implicit Team 3 strength would exceed its target.
                s3_current = current_total_sum_B - (s1 + s2)
                if s3_current <= target_strength:
                    cost = 1 if initial_team != 3 else 0
                    new_dp[s1][s2] = min(new_dp[s1][s2], dp[s1][s2] + cost)
        
        # Update dp to new_dp for the next person's processing.
        dp = new_dp
            
    # After processing all N people, the minimum moves to balance teams 1 and 2
    # to target_strength is stored in dp[target_strength][target_strength].
    # At this point, the total strength of all N people is `total_strength`.
    # If s1 = target_strength and s2 = target_strength, then
    # s3_final = total_strength - (target_strength + target_strength) = target_strength.
    # So, if dp[target_strength][target_strength] is reachable, all three teams are balanced.
    result = dp[target_strength][target_strength]
    
    # If the result is still INF, it means the target state is unreachable.
    if result == INF:
        print(-1)
    else:
        print(result)

solve()