import bisect
import sys

# Helper function to read a line of space-separated integers
def read_int_list():
    return list(map(int, sys.stdin.readline().split()))

# Helper function to read a single integer
def read_int():
    return int(sys.stdin.readline())

def main():
    N = read_int()
    X_coords = read_int_list()
    P_populations = read_int_list()

    # Calculate prefix sums of populations.
    # cumulative_populations[k] will store the sum of P_populations[0]...P_populations[k-1].
    # It has N+1 elements. cumulative_populations[0] is 0.
    cumulative_populations = [0] * (N + 1)
    # cumulative_populations[0] is already 0 from initialization.
    for i in range(N): # P_populations has N elements: P_populations[0] to P_populations[N-1]
        cumulative_populations[i+1] = cumulative_populations[i] + P_populations[i]

    Q_count = read_int()

    # Process each query
    for _ in range(Q_count):
        L, R = read_int_list()
        
        # Find the 0-based index of the first village X_k such that X_k >= L.
        # X_coords is sorted.
        start_idx = bisect.bisect_left(X_coords, L)
        
        # Find the 0-based index 'k' such that for all j < k, X_coords[j] <= R.
        # This is equivalent to finding the index of the first village X_k such that X_k > R.
        # The villages included are those with indices from start_idx to end_idx-1.
        end_idx = bisect.bisect_right(X_coords, R)
        
        # The sum of populations P_populations[start_idx] + ... + P_populations[end_idx-1]
        # is given by cumulative_populations[end_idx] - cumulative_populations[start_idx].
        total_villagers = cumulative_populations[end_idx] - cumulative_populations[start_idx]
        
        # Print the result for the current query. print() adds a newline automatically.
        print(total_villagers)

# Standard practice for Python scripts to be executable
if __name__ == '__main__':
    main()