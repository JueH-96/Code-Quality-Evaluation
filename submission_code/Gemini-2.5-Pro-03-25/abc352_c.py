import sys

def solve():
    # Read the number of giants, N
    N = int(sys.stdin.readline())
    
    # Initialize the total sum of shoulder heights
    sum_A = 0
    
    # Initialize the maximum difference between head height and shoulder height found so far.
    # The problem states A_i <= B_i, which implies B_i - A_i >= 0.
    # Therefore, initializing max_D with 0 is safe and correct. Any valid difference will be >= 0.
    max_D = 0 
    
    # Iterate through each of the N giants
    for _ in range(N):
        # Read the shoulder height A and head height B for the current giant.
        # sys.stdin.readline() reads a line from standard input.
        # .split() splits the line into parts based on whitespace.
        # map(int, ...) applies the int conversion function to each part.
        A, B = map(int, sys.stdin.readline().split())
        
        # Add the current giant's shoulder height A to the running total sum_A.
        sum_A += A
        
        # Calculate the difference D = B - A for the current giant.
        # This represents the height added by the head above the shoulders.
        D = B - A
        
        # Update max_D if the current giant's difference D is greater than the maximum difference found so far.
        # The max() function conveniently returns the larger of its two arguments.
        max_D = max(max_D, D)
            
    # The final height of the topmost giant's head is derived as follows:
    # Let the permutation be P = (P_1, ..., P_N).
    # The height of the shoulders of giant P_k is S(P_k) = sum_{j=1 to k} A_{P_j}.
    # The height of the head of giant P_k is H(P_k) = S(P_{k-1}) + B_{P_k} for k > 1, and H(P_1) = B_{P_1}.
    # We want to maximize H(P_N).
    # H(P_N) = S(P_{N-1}) + B_{P_N}
    # H(P_N) = (sum_{j=1 to N-1} A_{P_j}) + B_{P_N}
    # Let sum_A = sum_{i=1 to N} A_i. This is the total sum of shoulder heights of all giants.
    # Then sum_{j=1 to N-1} A_{P_j} = sum_A - A_{P_N}.
    # So, H(P_N) = (sum_A - A_{P_N}) + B_{P_N} = sum_A + (B_{P_N} - A_{P_N}).
    # To maximize H(P_N), we need to maximize B_{P_N} - A_{P_N}, since sum_A is constant regardless of the permutation.
    # The maximum value of B_{P_N} - A_{P_N} is max_D, which we calculated.
    # Therefore, the maximum possible height is sum_A + max_D.
    result = sum_A + max_D
    
    # Print the final calculated maximum height to standard output.
    print(result)

# This is a standard Python construct. It ensures that the solve() function is called 
# only when the script is executed directly (not when imported as a module).
if __name__ == '__main__':
    solve()