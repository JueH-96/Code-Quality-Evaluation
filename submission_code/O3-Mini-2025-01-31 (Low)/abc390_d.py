def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:]))

    # We want to count the distinct XOR values that can appear after 
    # merging according to some sequence of operations.
    # A merge operation moves all stones from one bag into another.
    # The final configuration is effectively a partition of the original
    # set of stone counts (each bag's value A_i) into several groups, where 
    # the final value in each nonempty bag is the sum of the A_i values in that group.
    # The final XOR is the XOR of these sums.
    #
    # Therefore, the problem reduces to: Given an array A of length N, 
    # consider every partition (every possible grouping of indices 0..N-1).
    # For each partition, compute XOR of (sum of A[i] for i in group) over groups.
    # Then count the number of distinct XOR values.
    #
    # Since N <= 12, the total number of partitions (Bell number of 12) is less than 500000,
    # so we can generate them using a recursion.

    result = set()

    # Backtracking function
    # i: current index we want to assign to a group.
    # groups: A list of groups so far; each group is represented as a list of indices.
    def backtrack(i, groups):
        if i == N:
            xor_val = 0
            # Compute XOR of sums in each group.
            for grp in groups:
                group_sum = 0
                for idx in grp:
                    group_sum += A[idx]
                xor_val ^= group_sum
            result.add(xor_val)
            return
        
        # Try to add A[i] to each existing group.
        for group in groups:
            group.append(i)
            backtrack(i + 1, groups)
            group.pop()
        
        # Try to start a new group with A[i].
        groups.append([i])
        backtrack(i + 1, groups)
        groups.pop()
    
    backtrack(0, [])
    sys.stdout.write(str(len(result)))

if __name__ == '__main__':
    main()