# YOUR CODE HERE
import sys

# Set higher recursion depth for safety, although this DP is iterative. Usually not needed for iterative DP.
# sys.setrecursionlimit(2000) 

def solve():
    # Read problem parameters N (number of wheels) and M (target points)
    N, M = map(int, sys.stdin.readline().split())
    
    # Store wheel data in a list of dictionaries
    wheels = []
    for i in range(N):
        # Read cost, number of outcomes, and the scores for each outcome
        line = list(map(int, sys.stdin.readline().split()))
        C = line[0] # Cost to play wheel i
        P = line[1] # Number of outcomes (integers written) for wheel i
        S = line[2:] # List of scores corresponding to the P outcomes of wheel i
        
        # Basic validation (optional, based on constraints or input format assumptions)
        # Example: assert P == len(S), "Number of outcomes P must match the length of the scores list S"
        # Example: assert sum(S) > 0, "Sum of scores for any wheel must be positive as per problem statement"
        
        wheels.append({'C': C, 'P': P, 'S': S})

    # Initialize DP table E
    # E[k] stores the minimum expected additional cost required to reach at least M points,
    # given that we currently have accumulated k points.
    # The table size is M+1 to cover states k = 0, 1, ..., M.
    E = [float('inf')] * (M + 1)
    
    # Base case: If we already have M or more points (represented by state M), 
    # the additional cost required is 0.
    E[M] = 0.0

    # DP calculation loop: Iterate backwards through states k from M-1 down to 0.
    # This order ensures that when computing E[k], all required values E[k'] for k' > k are already computed.
    for k in range(M - 1, -1, -1):
        
        # For the current state k, find the minimum expected cost by considering playing each wheel.
        # Initialize the minimum cost for state k to infinity.
        min_expected_cost_for_k = float('inf')
        
        # Iterate through each available wheel
        for i in range(N):
            wheel = wheels[i]
            C_i = wheel['C'] # Cost of playing wheel i
            P_i = wheel['P'] # Number of outcomes for wheel i
            S_i = wheel['S'] # List of scores for wheel i

            # Calculate necessary statistics for the expected cost formula for wheel i at state k
            count_positive_scores = 0 # This will store P_{i, >0}, the count of outcomes with score > 0
            sum_future_costs = 0.0 # This will store the sum of E[next_state] for all outcomes with score > 0
            
            # Iterate through all possible outcomes j = 1..P_i for the current wheel i
            for j in range(P_i):
                score = S_i[j] # The score obtained in outcome j
                
                # We only need to sum future costs for outcomes that result in earning positive points.
                # Outcomes with 0 points are handled implicitly by the derived formula.
                if score > 0:
                    count_positive_scores += 1
                    
                    # Determine the point total after this play. If it exceeds M, cap it at M,
                    # as all states >= M are equivalent (target reached).
                    next_state_points = min(k + score, M)
                    
                    # Add the expected cost from the resulting state (which is already computed) to the sum.
                    sum_future_costs += E[next_state_points]
            
            # P_i_gt0 represents the number of outcomes yielding > 0 points for wheel i.
            P_i_gt0 = count_positive_scores
            
            # The problem constraint states that the sum of scores for any wheel is positive.
            # This guarantees that there is at least one outcome with score > 0, so P_i_gt0 >= 1.
            # Therefore, division by P_i_gt0 is always safe (no division by zero).
            
            # Compute the expected cost V_i(k) if we choose to play wheel i when in state k.
            # The formula is derived by considering the cost C_i and the expected future costs,
            # properly handling the possibility of getting 0 points and staying in state k effectively.
            # Formula: V_i(k) = (P_i * C_i + sum_{j where S_{i,j}>0} E[min(k+S_{i,j}, M)]) / P_{i,>0}
            current_wheel_expected_cost = (P_i * C_i + sum_future_costs) / P_i_gt0
            
            # Update the minimum expected cost found so far for state k. We choose the wheel that minimizes this value.
            min_expected_cost_for_k = min(min_expected_cost_for_k, current_wheel_expected_cost)

        # After considering all wheels, store the computed minimum expected cost for state k in the DP table.
        E[k] = min_expected_cost_for_k

    # The final answer is the minimum expected cost starting from 0 points, which is E[0].
    # Print the result using floating point format with sufficient precision as required by the problem statement.
    print(f"{E[0]:.17f}")

# Execute the solve function to run the main logic of the program
solve()