def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    N = int(data[0])
    M = int(data[1])
    
    # Read sequence A and B from input
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    # Combine A and B then sort to form C
    C = sorted(A + B)
    
    # Create a set for quick membership test for elements from A
    set_A = set(A)
    
    # Check adjacent pairs in C to see if both are from A
    for i in range(len(C) - 1):
        if C[i] in set_A and C[i+1] in set_A:
            print("Yes")
            return
            
    print("No")

if __name__ == '__main__':
    main()