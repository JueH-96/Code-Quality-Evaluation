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
    people = [(A[i], i+1) for i in range(N)]
    # Sort the people based on A_i
    people.sort()
    
    # Extract sorted A and their original indices
    sorted_A = [p[0] for p in people]
    sorted_indices = [p[1] for p in people]
    
    # Initialize a list to keep track of the next available person for each sushi
    # We will use a list to mark whether a person has already taken a sushi
    taken = [False] * (N + 1)
    
    for b in B:
        # Find the first person whose A_i is <= b and is not taken
        # Using bisect_left to find the first A_i <= b
        idx = bisect.bisect_right(sorted_A, b)
        if idx == 0:
            # No person can take this sushi
            print(-1)
            continue
        # Iterate from the first possible person to the last
        found = False
        for i in range(idx-1, -1, -1):
            if not taken[sorted_indices[i]]:
                print(sorted_indices[i])
                taken[sorted_indices[i]] = True
                found = True
                break
        if not found:
            print(-1)

if __name__ == "__main__":
    main()