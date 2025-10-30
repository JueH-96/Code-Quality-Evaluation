import sys

def main():
    # Read the first line and split into N, L, R
    N_L_R = sys.stdin.readline().split()
    N = int(N_L_R[0])
    L = int(N_L_R[1])
    R = int(N_L_R[2])
    
    # Read the second line and split into the array A
    A = list(map(int, sys.stdin.readline().split()))
    
    # Determine X_i for each A_i and collect results as strings
    X = [str(L) if a < L else str(R) if a > R else str(a) for a in A]
    
    # Print the results separated by spaces
    print(' '.join(X))

if __name__ == "__main__":
    main()