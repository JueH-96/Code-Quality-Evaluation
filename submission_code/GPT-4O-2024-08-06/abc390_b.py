# YOUR CODE HERE
def is_geometric_progression(N, A):
    if N == 2:
        # Any two numbers can form a geometric progression
        return True
    
    # Calculate the common ratio using the first two elements
    r = A[1] / A[0]
    
    # Check if each subsequent pair of elements maintains the same ratio
    for i in range(1, N):
        if A[i] / A[i - 1] != r:
            return False
    
    return True

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    if is_geometric_progression(N, A):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()