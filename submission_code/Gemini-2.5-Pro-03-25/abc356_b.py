# YOUR CODE HERE
import sys

def solve():
    # Read N (number of foods) and M (number of nutrients) from the first line of input
    n, m = map(int, sys.stdin.readline().split())
    
    # Read the required minimum amount for each nutrient (A_1, ..., A_M)
    # We store them in a list `a`. Note that list indices are 0-based,
    # so `a[j]` corresponds to the requirement A_{j+1} in the problem statement.
    a = list(map(int, sys.stdin.readline().split()))
    
    # Initialize a list `total_nutrients` of size M with all elements set to 0.
    # `total_nutrients[j]` will store the total amount of nutrient j+1 consumed.
    total_nutrients = [0] * m
    
    # Loop through each of the N foods
    for _ in range(n):
        # Read the nutrient amounts provided by the current food (X_{i,1}, ..., X_{i,M})
        # `x_row[j]` contains the amount of nutrient j+1 from this food.
        x_row = list(map(int, sys.stdin.readline().split()))
        
        # Add the nutrients from this food to the running totals
        # This inner loop iterates through each nutrient type (0 to M-1)
        for j in range(m):
            total_nutrients[j] += x_row[j]
            
    # After processing all foods, check if the goal is met for every nutrient
    # Iterate through each nutrient type index j from 0 to M-1
    for j in range(m):
        # Compare the total intake `total_nutrients[j]` with the required amount `a[j]`
        # If the total intake for nutrient j+1 is less than the required amount A_{j+1}
        if total_nutrients[j] < a[j]:
            # If any nutrient goal is not met, print "No"
            print("No") 
            # Exit the function immediately as the overall condition is false
            return
            
    # If the loop completes without returning, it means the total intake met or exceeded
    # the required amount for all M nutrients.
    # In this case, print "Yes".
    print("Yes")

# Call the solve function to execute the logic
solve()