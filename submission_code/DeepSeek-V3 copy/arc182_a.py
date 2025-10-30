MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    P = []
    V = []
    index = 2
    for _ in range(Q):
        P.append(int(data[index]))
        V.append(int(data[index+1]))
        index += 2
    
    # Initialize the sequence S with 0
    S = [0] * N
    
    # We need to determine the order of operations that allows all operations to be performed without crying
    # The key is to ensure that for each operation, the value V_i is not less than any element in the range it affects
    
    # We can process the operations in a specific order to ensure that the conditions are met
    # One approach is to process the operations in the order of increasing V_i
    
    # Sort the operations based on V_i
    operations = sorted(zip(P, V), key=lambda x: x[1])
    
    # Now, for each operation, we need to determine if it can be performed without crying
    # We will simulate the operations in the sorted order
    
    # Initialize the sequence S with 0
    S = [0] * N
    
    # To keep track of the possible choices for each operation
    # For each operation, we have two choices: left or right
    # We need to count the number of valid sequences of choices
    
    # We can use dynamic programming to count the number of valid sequences
    # Let dp[i] represent the number of valid sequences up to the i-th operation
    
    # Initialize dp[0] = 1 (no operations have been performed yet)
    dp = [1]
    
    for i in range(Q):
        p, v = operations[i]
        # Determine the range affected by the operation
        # For the left operation: S_1 to S_p
        # For the right operation: S_p to S_N
        
        # Check if the left operation is possible
        left_possible = True
        for j in range(p):
            if S[j] > v:
                left_possible = False
                break
        
        # Check if the right operation is possible
        right_possible = True
        for j in range(p-1, N):
            if S[j] > v:
                right_possible = False
                break
        
        # Calculate the number of valid choices for this operation
        choices = 0
        if left_possible:
            choices += 1
        if right_possible:
            choices += 1
        
        # Update the dp array
        dp.append((dp[i] * choices) % MOD)
        
        # Update the sequence S based on the chosen operation
        # Since we are processing in order of increasing V_i, we can choose any valid operation
        # For simplicity, we choose the left operation if possible, otherwise the right operation
        if left_possible:
            for j in range(p):
                S[j] = v
        elif right_possible:
            for j in range(p-1, N):
                S[j] = v
        else:
            # If neither operation is possible, the sequence is invalid
            dp[-1] = 0
            break
    
    print(dp[Q] % MOD)

if __name__ == "__main__":
    main()