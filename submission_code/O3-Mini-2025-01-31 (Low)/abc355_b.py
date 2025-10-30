def main():
    import sys
    input_data = sys.stdin.read().split()
    # first two integers are N and M
    N = int(input_data[0])
    M = int(input_data[1])
    
    # next N values form sequence A
    A = list(map(int, input_data[2:2+N]))
    
    # next M values form sequence B
    B = list(map(int, input_data[2+N:2+N+M]))
    
    # Create a set for quick membership check for A
    set_A = set(A)
    
    # Sort all numbers together
    C = sorted(A + B)
    
    # Check for two consecutive elements both in A
    for i in range(len(C) - 1):
        if C[i] in set_A and C[i+1] in set_A:
            print("Yes")
            return
    print("No")
    
if __name__ == "__main__":
    main()