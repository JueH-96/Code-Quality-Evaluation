# YOUR CODE HERE
def find_second_largest_index():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Create a list of tuples (value, index) and sort it by value
    indexed_A = [(A[i], i + 1) for i in range(N)]
    indexed_A.sort(reverse=True, key=lambda x: x[0])
    
    # The second element in the sorted list is the second largest
    second_largest_index = indexed_A[1][1]
    
    print(second_largest_index)

find_second_largest_index()