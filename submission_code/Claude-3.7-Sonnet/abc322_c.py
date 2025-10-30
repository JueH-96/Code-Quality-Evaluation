# YOUR CODE HERE
def solve():
    # Read input
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Initialize the pointer to the first firework day
    firework_index = 0
    
    # Process each day of the festival
    for i in range(1, N+1):
        # Find the next firework day on or after the current day
        while firework_index < M and A[firework_index] < i:
            firework_index += 1
        
        # Calculate and print the number of days until the next firework
        print(A[firework_index] - i)

solve()