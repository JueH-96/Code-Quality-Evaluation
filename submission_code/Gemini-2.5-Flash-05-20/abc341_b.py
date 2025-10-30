import sys

def solve():
    # Read N, the number of countries
    N = int(sys.stdin.readline())
    
    # Read the initial amounts of currency for each country
    # A[0] corresponds to A_1, A[1] to A_2, ..., A[N-1] to A_N
    A = list(map(int, sys.stdin.readline().split()))
    
    # Read the conversion rates (S_i, T_i) for each operation
    # S_T_pairs[i] corresponds to (S_{i+1}, T_{i+1})
    # This means S_T_pairs[0] is (S_1, T_1), used for converting A_1 to A_2
    # and S_T_pairs[N-2] is (S_{N-1}, T_{N-1}), used for converting A_{N-1} to A_N
    S_T_pairs = []
    for _ in range(N - 1):
        s_i, t_i = map(int, sys.stdin.readline().split())
        S_T_pairs.append((s_i, t_i))
        
    # Iterate through countries from 1 to N-1 (0-indexed: from index 0 to N-2)
    # At each step 'i', we process conversions from country (i+1) to country (i+2)
    # using A[i] and A[i+1] values
    for i in range(N - 1):
        s_val, t_val = S_T_pairs[i] # Get S_i and T_i for current conversion (from A[i] to A[i+1])
        
        # Calculate how many times the operation can be performed.
        # This is limited by the current amount of currency in country (i+1) (A[i])
        # and the cost (s_val) for one operation.
        # Integer division automatically handles cases where A[i] < s_val (num_conversions will be 0).
        num_conversions = A[i] // s_val
        
        # Add the gained currency to the balance of the next country (i+2).
        # In 0-indexed terms, this means adding to A[i+1].
        A[i+1] += num_conversions * t_val
        
        # Note: The currency A[i] used for conversion is "paid" and removed from A[i].
        # However, for the purpose of finding the maximum A[N], the exact remaining
        # value of A[i] (A[i] % s_val) is irrelevant as it cannot be used for
        # further conversions affecting A[N] (operations only go forward).
        # Thus, there's no need to explicitly update A[i] (e.g., A[i] %= s_val).

    # After processing all possible conversions up to country N-1,
    # A[N-1] will contain the maximum possible number of units of currency of country N.
    print(A[N-1])

# Call the solve function to execute the program
solve()