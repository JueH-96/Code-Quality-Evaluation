import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    results = set()

    # current_groups is a list of sums of stones in merged bags.
    # Each element in current_groups represents the total stones in one
    # of the k non-empty bags in the current partition configuration.
    # idx is the index of the stone A[idx] from the input array A
    # that we are currently trying to place into a group.
    def solve(idx, current_groups):
        if idx == N:
            # Base case: all stones (A[0]...A[N-1]) have been placed into groups.
            # An actual configuration of bags is formed.
            # Calculate the XOR sum of stones in these groups.
            # Problem constraints state N >= 2. The first element A[0] must start a new group,
            # so current_groups will always have at least one group when idx > 0.
            # Thus, current_groups will not be empty here.
            
            xor_sum = 0
            for group_sum in current_groups:
                xor_sum ^= group_sum
            results.add(xor_sum)
            return

        # Recursive step: consider stone A[idx]
        
        # Option 1: Add A[idx] to one of the existing groups.
        # Iterate through each existing group, temporarily add A[idx] to it,
        # and recurse for the next stone A[idx+1].
        for i in range(len(current_groups)):
            current_groups[i] += A[idx]
            solve(idx + 1, current_groups)
            current_groups[i] -= A[idx] # Backtrack: remove A[idx] from this group to restore state.

        # Option 2: A[idx] starts a completely new group.
        # Add A[idx] as a new element to current_groups list,
        # and recurse for the next stone A[idx+1].
        current_groups.append(A[idx])
        solve(idx + 1, current_groups)
        current_groups.pop() # Backtrack: remove the new group started by A[idx] to restore state.

    # Initial call to the recursive function:
    # Start with the 0-th stone (A[0]) and an empty list of groups.
    # The first stone A[0] will always start a new group as current_groups is initially empty.
    solve(0, [])
    
    # Print the total number of distinct XOR sums found.
    print(len(results))

if __name__ == '__main__':
    main()