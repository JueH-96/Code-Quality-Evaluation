# YOUR CODE HERE
import sys
import bisect

def main():
    # Read N (number of sleighs) and Q (number of queries) from the first line of input
    N, Q = map(int, sys.stdin.readline().split())
    
    # Read the list of reindeer requirements R_1, ..., R_N for each sleigh
    R = list(map(int, sys.stdin.readline().split()))
    
    # Sort the reindeer requirements in non-decreasing order.
    # To maximize the number of sleighs we can pull with X reindeer,
    # it's always optimal to choose the sleighs with the smallest requirements first.
    R.sort()
    
    # Compute the prefix sums of the sorted requirements.
    # prefix_sums[k] will store the minimum total reindeer needed to pull k sleighs.
    # Specifically, prefix_sums[k] = R'_1 + R'_2 + ... + R'_k, where R' are the sorted requirements.
    # We use a list of size N+1 to store prefix sums from k=0 to k=N.
    # prefix_sums[0] = 0, representing the cost of pulling 0 sleighs.
    prefix_sums = [0] * (N + 1) 
    for i in range(N):
        # prefix_sums[i+1] = prefix_sums[i] + R[i]
        # R[i] is the i-th element of the sorted list (0-indexed), 
        # which corresponds to the (i+1)-th smallest requirement.
        prefix_sums[i+1] = prefix_sums[i] + R[i]
        # Python's integers handle arbitrary size, so overflow up to 2*10^14 is not an issue.
        
    # List to store the results for each query as strings
    results = []
    
    # Process each of the Q queries
    for _ in range(Q):
        # Read the available number of reindeer X for the current query
        X = int(sys.stdin.readline())
        
        # We need to find the maximum integer k (number of sleighs) such that 
        # the minimum cost to pull k sleighs, prefix_sums[k], is less than or equal to X.
        # Since the prefix_sums list is non-decreasing, we can efficiently find this k using binary search.
        
        # Python's bisect.bisect_right function performs binary search.
        # bisect_right(a, x) finds an insertion point `idx` such that all elements `a[i]` for `i < idx` 
        # satisfy `a[i] <= x`, and all elements `a[i]` for `i >= idx` satisfy `a[i] > x`.
        
        # Applying this to our prefix_sums array:
        # idx = bisect.bisect_right(prefix_sums, X) gives the index `idx`.
        # This means prefix_sums[k] <= X for all k < idx.
        # And prefix_sums[k] > X for all k >= idx.
        
        # Therefore, the maximum index `k` for which prefix_sums[k] <= X is `idx - 1`.
        # This maximum index `k` represents the maximum number of sleighs we can pull.
        idx = bisect.bisect_right(prefix_sums, X)
        
        # The answer for the query is k = idx - 1.
        ans = idx - 1
        
        # Convert the answer to string and add to the results list for later printing.
        results.append(str(ans))
        
    # Print all the results, each on a new line.
    # Using "
".join is generally efficient for printing multiple lines.
    print("
".join(results))

# The standard entry point for Python scripts.
if __name__ == '__main__':
    main()
# END OF YOUR CODE HERE