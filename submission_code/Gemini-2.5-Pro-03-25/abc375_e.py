# YOUR CODE HERE
import sys

# Function to read input and solve the problem
def solve():
    # Read N
    N = int(sys.stdin.readline())
    
    # Read N lines of A_i, B_i
    people = []
    total_strength = 0
    for i in range(N):
        line = sys.stdin.readline().split()
        A_i = int(line[0]) # Original team of person i
        B_i = int(line[1]) # Strength of person i
        # Store required info for each person
        people.append({'A': A_i, 'B': B_i}) 
        total_strength += B_i

    # Check if total strength is divisible by 3. If not, equal strength teams are impossible.
    if total_strength % 3 != 0:
        print("-1")
        return

    # Target strength for each team
    T = total_strength // 3
    
    # Calculate prefix sums of strengths. This helps efficiently calculate the strength 
    # of the third team implicitly defined by the strengths of the first two teams.
    # prefix_strengths[k] stores the sum of strengths of the first k people.
    strengths = [p['B'] for p in people]
    prefix_strengths = [0] * (N + 1)
    for i in range(N):
        prefix_strengths[i+1] = prefix_strengths[i] + strengths[i]

    # Dynamic Programming Approach:
    # State: dp[s1][s2] = minimum number of switches required among the people processed so far
    # such that team 1 has total strength s1 and team 2 has total strength s2.
    # The strength of team 3 is implicitly determined by the total strength processed so far minus s1 and s2.
    
    # Initialize DP table using a 2D list. The dimensions are (T+1) x (T+1).
    # Use infinity value larger than the maximum possible cost N. N+1 is sufficient.
    infinity = N + 1 
    
    # dp table stores results after considering person i-1. Initialize for 0 people processed.
    dp = [[infinity] * (T + 1) for _ in range(T + 1)]
    
    # Base case: After processing 0 people, team 1 has strength 0, team 2 has strength 0, with 0 switches.
    dp[0][0] = 0

    # Iterate through each person from 1 to N
    for i in range(N):
        # Current person's details
        person = people[i]
        B_i = person['B'] # Strength of person i
        A_i = person['A'] # Original team of person i
        
        # Total strength of people considered up to and including person i
        # This is prefix_strengths[i+1] because prefix_strengths is 1-indexed effectively (index 0 is 0 sum).
        current_S_i = prefix_strengths[i+1] 
        
        # Create a new DP table for results after considering person i. Initialize with infinity.
        # Using a new table prevents using person i multiple times within the same iteration's updates.
        new_dp = [[infinity] * (T + 1) for _ in range(T + 1)]
        
        # Iterate through all possible states (s1, s2) reachable after considering person i-1
        for s1 in range(T + 1):
            for s2 in range(T + 1):
                # If the state (s1, s2) was unreachable after processing the previous person (i-1), skip it.
                if dp[s1][s2] == infinity:
                    continue
                
                # Get the minimum cost found so far to reach state (s1, s2) using first i-1 people
                current_cost = dp[s1][s2]
                
                # Explore the three possibilities for assigning person i:
                
                # --- Option 1: Assign person i to Team 1 ---
                new_s1_t1 = s1 + B_i # New strength for team 1
                # Check if the new strength for team 1 does not exceed the target T
                if new_s1_t1 <= T:
                    # Calculate cost increase: 1 if person i switched teams (A_i != 1), 0 otherwise.
                    cost_increase = 0 if A_i == 1 else 1
                    new_cost = current_cost + cost_increase
                    # Update the minimum cost for the resulting state (new_s1_t1, s2) in the new_dp table.
                    new_dp[new_s1_t1][s2] = min(new_dp[new_s1_t1][s2], new_cost)
                
                # --- Option 2: Assign person i to Team 2 ---
                new_s2_t2 = s2 + B_i # New strength for team 2
                # Check if the new strength for team 2 does not exceed the target T
                if new_s2_t2 <= T:
                    # Calculate cost increase: 1 if person i switched teams (A_i != 2), 0 otherwise.
                    cost_increase = 0 if A_i == 2 else 1
                    new_cost = current_cost + cost_increase
                    # Update the minimum cost for the resulting state (s1, new_s2_t2) in the new_dp table.
                    new_dp[s1][new_s2_t2] = min(new_dp[s1][new_s2_t2], new_cost)

                # --- Option 3: Assign person i to Team 3 ---
                # If person i is assigned to team 3, strengths s1 and s2 remain unchanged from the state after i-1 people.
                # However, the implied strength of team 3 increases by B_i. We must check this constraint.
                # The strength of team 3 is the total strength processed so far (current_S_i) minus s1 and s2.
                s3 = current_S_i - s1 - s2
                
                # Check if the resulting strength for team 3 does not exceed the target T
                if s3 <= T:
                     # Calculate cost increase: 1 if person i switched teams (A_i != 3), 0 otherwise.
                     cost_increase = 0 if A_i == 3 else 1
                     new_cost = current_cost + cost_increase
                     # Update the minimum cost for state (s1, s2) in the new_dp table. This state (s1, s2) now reflects 
                     # the assignment of person i to team 3.
                     new_dp[s1][s2] = min(new_dp[s1][s2], new_cost)
                     
        # After iterating through all states from dp, update dp table for the next iteration (person i+1)
        # The new_dp table now contains the minimum costs after processing person i.
        dp = new_dp 

    # After processing all N people, the final answer is the minimum cost stored in dp[T][T].
    # This state corresponds to team 1 having strength T and team 2 having strength T.
    # Since the total strength is 3T, team 3 must also have strength T.
    final_cost = dp[T][T]

    # If dp[T][T] is still infinity, it means the target state (T, T) is unreachable.
    if final_cost == infinity:
        print("-1")
    else:
        # Otherwise, print the minimum number of switches found.
        print(final_cost)

# Standard check to run the solve function when the script is executed.
if __name__ == '__main__':
    solve()