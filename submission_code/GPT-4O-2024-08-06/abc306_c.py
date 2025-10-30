# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Dictionary to store the middle occurrence index for each number
    middle_occurrence = {}
    
    # Dictionary to store the occurrences of each number
    occurrences = {}
    
    for index, value in enumerate(A):
        if value not in occurrences:
            occurrences[value] = []
        occurrences[value].append(index + 1)  # Store 1-based index
    
    for i in range(1, N + 1):
        # Get the middle occurrence index for each number
        middle_occurrence[i] = occurrences[i][1]
    
    # Sort numbers 1 to N based on their middle occurrence index
    sorted_numbers = sorted(range(1, N + 1), key=lambda x: middle_occurrence[x])
    
    # Print the sorted numbers
    print(" ".join(map(str, sorted_numbers)))

if __name__ == "__main__":
    main()