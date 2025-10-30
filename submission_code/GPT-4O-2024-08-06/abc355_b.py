# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    # Combine A and B
    C = A + B
    
    # Sort C
    C.sort()
    
    # Check for consecutive elements in C that are both from A
    a_set = set(A)
    
    for i in range(len(C) - 1):
        if C[i] in a_set and C[i+1] in a_set:
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()