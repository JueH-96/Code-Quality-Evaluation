# YOUR CODE HERE
def minimum_trip_cost():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = int(data[1])
    P = int(data[2])
    F = list(map(int, data[3:]))
    
    # Calculate the total cost if no passes are used
    total_regular_cost = sum(F)
    
    # Sort fares in descending order
    F.sort(reverse=True)
    
    # Initialize the minimum cost as the total regular cost
    min_cost = total_regular_cost
    
    # Calculate the cost using passes
    current_pass_cost = 0
    for i in range(0, N, D):
        # Add the cost of one more batch of passes
        current_pass_cost += P
        
        # Calculate the savings by using passes for the next D days
        # or the remaining days if less than D
        savings = sum(F[i:i+D])
        
        # Calculate the new total cost
        new_cost = total_regular_cost - savings + current_pass_cost
        
        # Update the minimum cost
        min_cost = min(min_cost, new_cost)
    
    print(min_cost)