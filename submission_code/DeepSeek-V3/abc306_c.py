# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:3*N+1]))
    
    # Create a dictionary to store the indices of each number
    indices = {}
    for idx, num in enumerate(A, 1):
        if num not in indices:
            indices[num] = []
        indices[num].append(idx)
    
    # Extract the middle index for each number
    f = {}
    for num in indices:
        f[num] = indices[num][1]  # Middle index is the second one (0-based index 1)
    
    # Sort the numbers based on their middle index
    sorted_nums = sorted(range(1, N+1), key=lambda x: f[x])
    
    # Print the result
    print(' '.join(map(str, sorted_nums)))

if __name__ == "__main__":
    main()