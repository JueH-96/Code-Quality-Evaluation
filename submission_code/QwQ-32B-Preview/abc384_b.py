import sys

def main():
    # Read N and R from the first line
    N_R = sys.stdin.readline().strip().split()
    N = int(N_R[0])
    R = int(N_R[1])
    
    # Initialize current rating
    current_rating = R
    
    # Process each of the N contests
    for _ in range(N):
        # Read D and A from the next line
        D_A = sys.stdin.readline().strip().split()
        D = int(D_A[0])
        A = int(D_A[1])
        
        # Update rating based on division and rating range
        if D == 1:
            if 1600 <= current_rating <= 2799:
                current_rating += A
        elif D == 2:
            if 1200 <= current_rating <= 2399:
                current_rating += A
    
    # Print the final rating
    print(current_rating)

if __name__ == '__main__':
    main()