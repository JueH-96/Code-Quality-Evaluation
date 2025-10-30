# YOUR CODE HERE
def find_second_largest_index():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Find the second largest element
    first_largest = second_largest = float('-inf')
    
    for number in A:
        if number > first_largest:
            second_largest = first_largest
            first_largest = number
        elif number > second_largest:
            second_largest = number
    
    # Find the index of the second largest element
    second_largest_index = A.index(second_largest) + 1  # +1 for 1-based index
    
    print(second_largest_index)

find_second_largest_index()