# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Dictionary to store the middle occurrence index of each number
    middle_indices = {}
    
    # Dictionary to count occurrences of each number
    count = {}
    
    for index, value in enumerate(A):
        if value not in count:
            count[value] = []
        count[value].append(index + 1)
    
    for key in count:
        middle_indices[key] = count[key][1]
    
    # Sort numbers 1 to N based on their middle occurrence index
    sorted_numbers = sorted(range(1, N + 1), key=lambda x: middle_indices[x])
    
    print(" ".join(map(str, sorted_numbers)))

if __name__ == "__main__":
    main()