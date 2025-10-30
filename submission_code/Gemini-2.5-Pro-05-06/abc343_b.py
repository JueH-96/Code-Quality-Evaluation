# YOUR CODE HERE
def main():
    N = int(input())

    adj_matrix = []
    for _ in range(N):
        # Each row is read as a list of integers
        # input().split() creates a list of strings
        # map(int, ...) converts these strings to integers
        row = list(map(int, input().split()))
        adj_matrix.append(row)

    # Iterate through each vertex, whose 0-indexed representation is 'i'
    # This corresponds to vertex 'i+1' in 1-indexed terms
    for i in range(N):
        neighbors = []
        # Iterate through all possible connections for vertex 'i+1'
        # 'j' is the 0-indexed representation of a potential neighbor vertex 'j+1'
        for j in range(N):
            # adj_matrix[i][j] == 1 means vertex 'i+1' is connected to vertex 'j+1'
            if adj_matrix[i][j] == 1:
                neighbors.append(j + 1)  # Store the 1-indexed neighbor label
        
        # The 'neighbors' list is already sorted because 'j' iterates from 0 to N-1,
        # so 'j+1' values are considered in ascending order.
        
        # Convert neighbor numbers to strings to use ' '.join()
        # If 'neighbors' is empty, map(str, neighbors) will be an empty iterator,
        # and ' '.join on an empty iterator results in an empty string.
        # Printing an empty string produces a blank line, as required.
        print(' '.join(map(str, neighbors)))

if __name__ == '__main__':
    main()