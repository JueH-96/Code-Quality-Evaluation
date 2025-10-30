# YOUR CODE HERE
def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    # Count how many odd values in the array
    odd_count = sum(1 for a in A if a % 2 == 1)
    
    if odd_count % 2 == 0:  # If even number of odd values
        if N % 2 == 1:  # If N is odd
            return "Fennec"
        else:  # If N is even
            return "Snuke"
    else:  # If odd number of odd values
        if N % 2 == 1:  # If N is odd
            return "Snuke"
        else:  # If N is even
            return "Fennec"

print(solve())