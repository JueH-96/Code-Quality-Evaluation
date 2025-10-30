import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    # Create a list of tuples (A_i, i+1) to keep track of the original indices
    A_with_indices = [(A[i], i+1) for i in range(N)]
    # Sort the list based on A_i
    A_with_indices.sort()
    # Extract the sorted A and the corresponding indices
    sorted_A = [a for a, idx in A_with_indices]
    sorted_indices = [idx for a, idx in A_with_indices]
    
    # Initialize a list to keep track of the first available index for each A_i
    # We will use a list to simulate the availability of each person
    # Initially, all are available
    available = [True] * N
    
    # For each sushi, find the first available person whose A_i is <= B_j
    for b in B:
        # Find the rightmost index where sorted_A[i] <= b
        idx = bisect.bisect_right(sorted_A, b) - 1
        if idx < 0:
            print(-1)
            continue
        # Find the first available person in the sorted list up to idx
        found = False
        for i in range(idx, -1, -1):
            if available[i]:
                print(sorted_indices[i])
                available[i] = False
                found = True
                break
        if not found:
            print(-1)

if __name__ == "__main__":
    main()