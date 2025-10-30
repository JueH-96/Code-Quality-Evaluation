def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    # Combine and sort the arrays
    C = sorted(A + B)
    
    # Set of elements in A for quick lookup
    A_set = set(A)
    
    # Check for consecutive elements from A in C
    for i in range(1, len(C)):
        if C[i] in A_set and C[i-1] in A_set:
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()