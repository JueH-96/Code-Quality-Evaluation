def main():
    # Read input
    N_T_A = input().split()
    N = int(N_T_A[0])
    T = int(N_T_A[1])
    A = int(N_T_A[2])
    
    # Calculate remaining votes
    R = N - (T + A)
    
    # Check if outcome is already decided
    if T > A + R or A > T + R:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()