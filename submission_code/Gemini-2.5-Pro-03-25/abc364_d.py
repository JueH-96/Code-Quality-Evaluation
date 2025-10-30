# YOUR CODE HERE
import sys
import bisect

# Function to read integers faster from standard input
def read_ints():
    """ Reads a line of space-separated integers from stdin """
    return map(int, sys.stdin.readline().split())

# Function to read a list of integers faster from standard input
def read_int_list():
    """ Reads a line of space-separated integers from stdin and returns as list """
    return list(map(int, sys.stdin.readline().split()))

def solve():
    """ Solves the main problem: finding the k-th smallest distance for Q queries """
    
    # Read N (number of A points) and Q (number of queries)
    N, Q = read_ints()
    
    # Read the coordinates of points A_1, ..., A_N
    A = read_int_list()
    
    # Sort the coordinates of A points. This allows efficient counting using binary search later.
    # The sorting takes O(N log N) time.
    A.sort() 

    results = [] # List to store the answer for each query

    # Process each of the Q queries
    for _ in range(Q):
        # Read the coordinate b_j of point B_j and the rank k_j
        b, k = read_ints()
        
        # We need to find the k_j-th smallest distance among |a_i - b_j| for i=1..N.
        # This is equivalent to finding the minimum distance D such that at least k_j points A_i
        # satisfy |a_i - b_j| <= D.
        # The condition |a_i - b_j| <= D is equivalent to a_i being in the interval [b_j - D, b_j + D].
        # We can use binary search on the possible values of the distance D.

        # The minimum possible distance is 0.
        low = 0 
        
        # The maximum possible distance occurs between the points with minimum and maximum possible coordinates.
        # Min coordinate is -10^8, max is 10^8. Max distance is 10^8 - (-10^8) = 2 * 10^8.
        high = 2 * 10**8 
        
        # Initialize the answer variable 'ans' to a value larger than any possible distance.
        # The binary search will update this value to the smallest valid distance found.
        ans = high + 1 

        # Perform binary search on the distance D in the range [low, high].
        # Each iteration takes O(log N) time due to bisect calls.
        # The number of iterations is O(log(MaxDistance)).
        # Total time per query: O(log(MaxDistance) * log N)
        while low <= high:
            # Calculate the middle value for the distance D
            mid = (low + high) // 2
            
            # Count the number of points A_i whose coordinates fall within the interval [b - mid, b + mid].
            # We use the pre-sorted list A and the bisect module for efficiency.
            
            # Find the index of the first element in A that is greater than or equal to (b - mid).
            # This is the lower bound of the range in the sorted list A.
            left_idx = bisect.bisect_left(A, b - mid)
            
            # Find the index of the first element in A that is strictly greater than (b + mid).
            # This index marks the end of elements that are <= (b + mid).
            # bisect_right returns this index.
            right_idx = bisect.bisect_right(A, b + mid)
            
            # The number of elements in A within the range [b - mid, b + mid] is right_idx - left_idx.
            count = right_idx - left_idx
            
            # Check if the count of points within distance 'mid' is sufficient.
            if count >= k:
                # If we found at least k points, then 'mid' is a possible distance value.
                # It could be the k-th smallest distance, or larger.
                # We store 'mid' as the current best answer (minimum valid distance found so far).
                # Then, we try to find an even smaller valid distance by searching in the lower half: [low, mid - 1].
                ans = mid 
                high = mid - 1 # Try smaller distances
            else:
                # If we found fewer than k points, the distance 'mid' is too small.
                # The k-th smallest distance must be larger than 'mid'.
                # We search in the upper half: [mid + 1, high].
                low = mid + 1
        
        # After the binary search loop terminates, 'ans' holds the minimum distance D
        # such that at least k points are within distance D from b. This value is exactly
        # the k-th smallest distance.
        results.append(ans)

    # Print all collected results, each on a new line.
    # Using '
'.join(map(str, results)) is generally faster than printing line by line in a loop
    # for a large number of results.
    print('
'.join(map(str, results)))

# Execute the solve function when the script is run
if __name__ == '__main__':
    solve()