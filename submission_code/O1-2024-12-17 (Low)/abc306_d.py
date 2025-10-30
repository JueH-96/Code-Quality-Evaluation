def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    # We'll use two variables to keep track of the DP state:
    # dpH = the maximum sum of tastiness achievable ending with a healthy stomach
    # dpU = the maximum sum of tastiness achievable ending with an upset stomach
    # If it's not possible to be in a certain state, we'll set its value to -inf.
    
    INF_NEG = float('-inf')
    
    dpH = 0          # Initially healthy with sum 0 (haven't eaten anything)
    dpU = INF_NEG    # Can't start upset
    
    idx = 1
    for _ in range(N):
        X = int(input_data[idx]); Y = int(input_data[idx+1])
        idx += 2
        
        newH = dpH   # Skip eating: remain the same if healthy
        newU = dpU   # Skip eating: remain the same if upset
        
        if X == 0:
            # If it's an antidote course:
            # eating while healthy remains healthy
            if dpH != INF_NEG:
                newH = max(newH, dpH + Y)
            # eating while upset returns to healthy
            if dpU != INF_NEG:
                newH = max(newH, dpU + Y)
        else:
            # If it's a poisonous course:
            # eating while healthy changes to upset
            if dpH != INF_NEG:
                newU = max(newU, dpH + Y)
            # eating while upset -> dead (invalid), so no transition
        
        dpH, dpU = newH, newU
    
    # We are alive if we end up either healthy or upset; take the maximum
    print(max(dpH, dpU))

# Call the main function
if __name__ == "__main__":
    main()