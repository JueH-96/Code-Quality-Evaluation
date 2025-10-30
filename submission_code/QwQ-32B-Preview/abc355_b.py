import sys

def main():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())
    
    # Read sequence A
    A = list(map(int, sys.stdin.readline().split()))
    
    # Read sequence B
    B = list(map(int, sys.stdin.readline().split()))
    
    # Create a set of A's elements for quick lookup
    A_set = set(A)
    
    # Combine A and B and sort to form sequence C
    C = sorted(A + B)
    
    # Iterate through C and check for consecutive elements from A
    for i in range(len(C) - 1):
        if C[i] in A_set and C[i + 1] in A_set:
            print("Yes")
            sys.exit()
    
    # If no such pair is found
    print("No")

if __name__ == "__main__":
    main()