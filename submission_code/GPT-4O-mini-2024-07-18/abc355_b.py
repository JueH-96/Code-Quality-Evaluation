# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and M
    N, M = map(int, data[0].split())
    
    # Read sequences A and B
    A = list(map(int, data[1].split()))
    B = list(map(int, data[2].split()))
    
    # Create the combined sequence C
    C = A + B
    C.sort()
    
    # Check for consecutive elements from A in the sorted list C
    A_set = set(A)
    for i in range(len(C) - 1):
        if C[i] in A_set and C[i + 1] in A_set:
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()