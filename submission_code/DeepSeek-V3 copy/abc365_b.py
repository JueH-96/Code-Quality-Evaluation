# YOUR CODE HERE
def find_second_largest_index():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Pair each element with its original index (1-based)
    indexed_A = [(A[i], i+1) for i in range(N)]
    
    # Sort in descending order based on the value
    sorted_A = sorted(indexed_A, key=lambda x: x[0], reverse=True)
    
    # The second largest element is at index 1 in the sorted list
    second_largest_index = sorted_A[1][1]
    
    print(second_largest_index)

find_second_largest_index()