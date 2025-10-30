import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    if N == 2:
        print("Yes")
        return
    
    # Calculate the common ratio from the first two elements
    if A[0] == 0 or A[1] == 0:
        print("No")
        return
    
    # We use integer division to check ratios to avoid floating point inaccuracies
    # Check if A[1] * A[0] == A[0] * A[2] to ensure integer ratio consistency
    if A[1] % A[0] != 0:
        print("No")
        return
    
    r = A[1] // A[0]
    
    for i in range(2, N):
        if A[i] % A[i-1] != 0 or (A[i] // A[i-1]) != r:
            print("No")
            return
    
    print("Yes")

main()