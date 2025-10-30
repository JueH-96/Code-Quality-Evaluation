import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Dictionary to store the indices of occurrences
    occurrences = {i: [] for i in range(1, N+1)}
    
    # Collect all occurrences of each number
    for index, value in enumerate(A):
        occurrences[value].append(index)
    
    # List to store (middle_index, number)
    middle_indices = []
    
    # Extract the middle occurrence for each number
    for number in range(1, N+1):
        middle_index = occurrences[number][1]  # The second occurrence (0-based index)
        middle_indices.append((middle_index, number))
    
    # Sort by the middle occurrence index
    middle_indices.sort()
    
    # Extract the sorted numbers
    sorted_numbers = [num for _, num in middle_indices]
    
    # Print the result
    print(" ".join(map(str, sorted_numbers)))

if __name__ == "__main__":
    main()