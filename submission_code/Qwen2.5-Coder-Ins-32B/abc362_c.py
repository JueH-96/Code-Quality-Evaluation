# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    pairs = [(int(input[2*i+1]), int(input[2*i+2])) for i in range(N)]
    
    # Initialize the sequence and the sum
    X = [0] * N
    current_sum = 0
    
    # Try to construct the sequence
    for i in range(N-1):
        L, R = pairs[i]
        # Assign the maximum possible value to X[i]
        X[i] = R
        current_sum += R
    
    # For the last element, we need to make the sum zero
    L_last, R_last = pairs[-1]
    required = -current_sum
    
    if L_last <= required <= R_last:
        X[-1] = required
        print("Yes")
        print(" ".join(map(str, X)))
    else:
        print("No")

if __name__ == "__main__":
    main()