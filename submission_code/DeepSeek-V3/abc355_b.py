# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    # Combine and sort
    C = sorted(A + B)
    
    # Create a set for A for quick lookup
    A_set = set(A)
    
    # Check for consecutive elements in A
    for i in range(len(C) - 1):
        if C[i] in A_set and C[i+1] in A_set:
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()