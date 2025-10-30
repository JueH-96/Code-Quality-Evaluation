# YOUR CODE HERE
import sys
# Using heapq could be an optimization but sorting is simpler and likely sufficient given total N constraint
# import heapq 

# Function to solve a single test case
def solve():
    # Read N (number of boxes) and M (number of ball types) for the current test case
    N, M = map(int, sys.stdin.readline().split())
    
    # Read box details (capacity V and price P)
    boxes_input = []
    for _ in range(N):
        boxes_input.append(list(map(int, sys.stdin.readline().split())))

    # Filter boxes to find those potentially profitable.
    # A box is potentially profitable only if its capacity V is strictly greater than its price P.
    # Store these valid boxes along with their potential profit W = V - P.
    valid_boxes = []
    for i in range(N):
        V, P = boxes_input[i]
        # Only consider boxes where capacity V > price P
        if V > P:
            # Store relevant info (V, P, and calculated profit W) as a dictionary
            # This helps in sorting based on different keys later.
            valid_boxes.append({'V': V, 'P': P, 'W': V - P})

    # Get the count S of valid boxes found.
    S = len(valid_boxes)
    
    # If there are no boxes that can yield a profit (S=0), 
    # Mr. Box cannot increase his money. The final money increase is 0.
    if S == 0:
        print(0)
        return

    # The optimal strategy depends on the relationship between M (ball types) and N (total boxes).
    # We analyze the game based on which player has a strategic advantage due to resource constraints.
    
    if M > N:
        # Case 1: Mr. Ball has more ball types (M) than the total number of boxes (N).
        # Mr. Ball can exploit this by introducing more distinct ball types than Mr. Box can handle with N boxes.
        # Specifically, Mr. Ball can force the game to end after Mr. Box uses k = min(N, S) boxes.
        # Mr. Ball gives one ball of type 1, Mr. Box uses box B1. Ball type 2, box B2... Ball type k, box Bk.
        # Then Mr. Ball gives ball type k+1 (if k<N) or N+1 (if k=N). Mr. Box has no available box for this new type and cannot buy more. Game ends.
        # Total revenue is k (1 yen per ball). Total cost is sum of prices of k boxes used. Profit = k - sum(P_i).
        # Mr. Box anticipates this. To maximize this profit, he must choose the k boxes from the valid set S
        # that have the minimum total price.
        
        # Determine the actual number of boxes Mr. Box will use, limited by N and S.
        k = min(N, S) 
        
        # If k is 0 (e.g., N=0 or S=0), profit is 0.
        # This check is slightly redundant due to the S==0 check earlier, but included for completeness.
        if k == 0:
             print(0)
             return

        # To find the k boxes with minimum prices, sort the valid boxes based on price P in ascending order.
        valid_boxes.sort(key=lambda x: x['P'])
        
        # Select the first k boxes from the sorted list; these have the minimum prices.
        min_price_boxes = valid_boxes[:k]
        
        # Calculate the total cost by summing the prices of these k selected boxes.
        total_cost = 0
        for box in min_price_boxes:
            total_cost += box['P']
            
        # Calculate the profit based on the strategy: k balls placed (revenue k) minus total cost.
        profit = k - total_cost
        
        # The final profit cannot be negative. If the calculation yields a negative value,
        # it means Mr. Box is better off not playing, resulting in 0 profit.
        print(max(0, profit))

    else: # M <= N
        # Case 2: Mr. Box has at least as many potentially available boxes (N) as there are ball types (M).
        # Mr. Box can potentially use up to M boxes, one for each ball type.
        # Mr. Ball wants to minimize the profit. He can do this by identifying the box Mr. Box chose
        # which yields the minimum profit (W = V - P), let's say box j, and then focusing on filling only that box.
        # Once box j is full (V_j balls placed), Mr. Ball offers one more ball of type j. Game ends.
        # The profit in this scenario is V_j - P_j = W_j.
        # Mr. Box anticipates Mr. Ball will force the profit to be the minimum W among the chosen boxes.
        # Therefore, Mr. Box chooses k = min(M, S) boxes from the valid set S that have the highest W values.
        # By doing this, Mr. Box maximizes the minimum guaranteed profit. The final profit will be the k-th largest W value.
        
        # Determine the actual number of boxes Mr. Box will use, limited by M and S.
        k = min(M, S)

        # If k becomes 0 (e.g., M=0 or S=0), profit is 0.
        if k == 0:
            print(0)
            return

        # To find the k boxes with maximum profits, sort the valid boxes based on profit W in descending order.
        valid_boxes.sort(key=lambda x: x['W'], reverse=True)
        
        # The k boxes with the highest profits are the first k elements in this sorted list.
        # The minimum profit among these top k boxes is determined by the k-th box (at index k-1).
        # This is because Mr. Ball will target this box to minimize the final outcome for Mr. Box.
        kth_profit = valid_boxes[k-1]['W']
        
        # The final profit cannot be negative.
        print(max(0, kth_profit))


# Read the number of test cases from standard input
T = int(sys.stdin.readline())
# Iterate through each test case and call the solve function
for _ in range(T):
    solve()