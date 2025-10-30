import math

def solve():
    N = int(input())
    people_data = []
    total_strength = 0
    for _ in range(N):
        a, b = map(int, input().split())
        people_data.append({'initial_team': a, 'strength': b})
        total_strength += b

    if total_strength % 3 != 0:
        print("-1")
        return

    target_strength_each_team = total_strength // 3
    X = target_strength_each_team

    # dp[s1][s2] = min switches to get team1 strength s1, team2 strength s2
    # Initialize dp table with infinity
    # Dimensions: (X+1) x (X+1)
    dp = [[math.inf] * (X + 1) for _ in range(X + 1)]
    dp[0][0] = 0  # Base case: 0 strength for T1, 0 for T2, 0 people considered, 0 switches.
    
    current_sum_strengths_processed = 0

    # Iterate through each person
    for person_idx in range(N):
        person = people_data[person_idx]
        b_i = person['strength'] # Strength of current person
        a_i = person['initial_team'] # Initial team of current person (1, 2, or 3)
        
        # Sum of strengths of people considered so far, *including* the current person
        current_sum_strengths_processed += b_i 
        
        # dp_new will store results after considering the current person
        dp_new = [[math.inf] * (X + 1) for _ in range(X + 1)]

        # Iterate over states achievable with first 'person_idx' people (i.e., people 0 to person_idx-1)
        # These states are stored in the 'dp' table from the previous iteration.
        for s1_prev in range(X + 1): 
            for s2_prev in range(X + 1): 
                if dp[s1_prev][s2_prev] == math.inf: # If this state was not reachable
                    continue
                
                cost_so_far = dp[s1_prev][s2_prev]

                # Option 1: Current person (person_idx) goes to team 1
                s1_curr_opt1 = s1_prev + b_i
                s2_curr_opt1 = s2_prev
                if s1_curr_opt1 <= X: # Team 1 strength does not exceed target
                    switches_opt1 = 0 if a_i == 1 else 1
                    dp_new[s1_curr_opt1][s2_curr_opt1] = min(dp_new[s1_curr_opt1][s2_curr_opt1], cost_so_far + switches_opt1)

                # Option 2: Current person (person_idx) goes to team 2
                s1_curr_opt2 = s1_prev
                s2_curr_opt2 = s2_prev + b_i
                if s2_curr_opt2 <= X: # Team 2 strength does not exceed target
                    switches_opt2 = 0 if a_i == 2 else 1
                    dp_new[s1_curr_opt2][s2_curr_opt2] = min(dp_new[s1_curr_opt2][s2_curr_opt2], cost_so_far + switches_opt2)
                
                # Option 3: Current person (person_idx) goes to team 3
                s1_curr_opt3 = s1_prev # Team 1 strength is unchanged by this person
                s2_curr_opt3 = s2_prev # Team 2 strength is unchanged by this person
                
                # Strength of team 3, after current person is added to it.
                # current_sum_strengths_processed includes strengths of people from index 0 to person_idx.
                # s1_prev and s2_prev are sums from people 0 to person_idx-1.
                # So, team 3 strength = current_sum_strengths_processed - s1_prev - s2_prev.
                s3_curr_opt3 = current_sum_strengths_processed - s1_prev - s2_prev
                
                if s3_curr_opt3 <= X: # Team 3 strength does not exceed target
                                      # s3_curr_opt3 >= 0 is guaranteed because:
                                      # s1_prev + s2_prev <= sum of strengths of people 0 to person_idx-1
                                      # current_sum_strengths_processed = (sum of strengths of people 0 to person_idx-1) + b_i
                                      # s3_curr_opt3 = (sum of strengths of people 0 to person_idx-1 - s1_prev - s2_prev) + b_i
                                      # Since the term in parenthesis is >=0 and b_i >=1, s3_curr_opt3 >= 1.
                    switches_opt3 = 0 if a_i == 3 else 1
                    dp_new[s1_curr_opt3][s2_curr_opt3] = min(dp_new[s1_curr_opt3][s2_curr_opt3], cost_so_far + switches_opt3)
        
        dp = dp_new # Update dp table to reflect results after considering current person

    ans = dp[X][X] # Final answer is when team1=X, team2=X (so team3 must also be X)

    if ans == math.inf:
        print("-1")
    else:
        print(ans)

solve()