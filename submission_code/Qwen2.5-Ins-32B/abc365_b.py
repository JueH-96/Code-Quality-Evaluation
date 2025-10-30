def find_second_largest_index():
    N = int(input())
    A = list(map(int, input().split()))
    
    # Create a list of tuples (value, original index)
    indexed_A = [(value, index + 1) for index, value in enumerate(A)]
    
    # Sort the list based on the value in descending order
    sorted_A = sorted(indexed_A, key=lambda x: x[0], reverse=True)
    
    # The second largest element will be at index 1 in the sorted list
    second_largest_index = sorted_A[1][1]
    
    print(second_largest_index)

find_second_largest_index()