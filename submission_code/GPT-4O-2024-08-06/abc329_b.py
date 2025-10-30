# YOUR CODE HERE
def find_second_largest():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Find the largest number
    largest = max(A)
    
    # Filter out the largest number and find the maximum of the rest
    filtered = [x for x in A if x != largest]
    second_largest = max(filtered)
    
    print(second_largest)