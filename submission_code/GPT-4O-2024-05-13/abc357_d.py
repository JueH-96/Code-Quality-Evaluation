# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    
    MOD = 998244353
    
    # Convert N to string
    N_str = str(N)
    
    # Create V_N by repeating N_str N times
    V_N_str = N_str * N
    
    # Convert V_N_str to integer and take modulo
    V_N = int(V_N_str) % MOD
    
    # Print the result
    print(V_N)

if __name__ == "__main__":
    main()