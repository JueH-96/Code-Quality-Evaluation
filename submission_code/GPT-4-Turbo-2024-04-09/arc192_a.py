# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    if all(A):
        print("Yes")
        return
    
    # We need to find at least one sequence of 0s that can be turned into 1s using the operations
    # We need to find a sequence of two consecutive 0s that can be targeted by either operation
    # Since operations can wrap around, we treat the array as circular
    
    # Check for two consecutive 0s in the array
    for i in range(N):
        if A[i] == 0 and A[(i + 1) % N] == 0:
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()