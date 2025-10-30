# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    
    # Calculate the total cost using the sum of the first N natural numbers
    total_cost = N * (N + 1) // 2
    
    # Output the total cost
    print(total_cost)