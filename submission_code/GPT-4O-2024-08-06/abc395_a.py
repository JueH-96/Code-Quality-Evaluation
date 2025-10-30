# YOUR CODE HERE
def is_strictly_increasing():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    for i in range(N - 1):
        if A[i] >= A[i + 1]:
            print("No")
            return
    
    print("Yes")

is_strictly_increasing()