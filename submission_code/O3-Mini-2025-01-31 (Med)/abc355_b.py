def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:2+N+M]))
    
    # Sort the combined list
    C = sorted(A + B)
    
    # Use a set for quick lookup to check membership in A
    A_set = set(A)
    
    # Check for any two consecutive elements in sorted list that both are in A
    for i in range(len(C) - 1):
        if C[i] in A_set and C[i+1] in A_set:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()