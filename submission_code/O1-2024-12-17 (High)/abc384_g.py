def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parse input
    N = int(input_data[0])
    A = list(map(int, input_data[1:1+N]))
    B = list(map(int, input_data[1+N:1+2*N]))
    K = int(input_data[1+2*N])
    
    # Queries start at index qpos in input_data
    qpos = 1 + 2*N + 1
    queries = []
    for i in range(K):
        x = int(input_data[qpos + 2*i])
        y = int(input_data[qpos + 2*i + 1])
        queries.append((x, y))
    
    # Prefix sums for A and B (for quick sum of first x or y elements)
    # We'll make them 1-indexed for convenience: A_sum[i] = sum of A[0..i-1]
    A_sum = [0]*(N+1)
    B_sum = [0]*(N+1)
    for i in range(1, N+1):
        A_sum[i] = A_sum[i-1] + A[i-1]
        B_sum[i] = B_sum[i-1] + B[i-1]
    
    def sum_of_min(sorted_a, sorted_b):
        """
        Given two sorted lists (ascending), return sum of min(a_i, b_j) 
        over all pairs (i,j) in the Cartesian product.
        
        Two-pointer trick for sum of pairwise minima:
          i, j start at 0
          If sorted_a[i] < sorted_b[j], then for all b_j' >= sorted_b[j],
             min(sorted_a[i], b_j') = sorted_a[i].
             There are (len(sorted_b) - j) such b_j's.
             Add sorted_a[i] * (len(sorted_b) - j) to the result, i += 1
          Else do the symmetric action with sorted_b[j].
        """
        i = j = 0
        la, lb = len(sorted_a), len(sorted_b)
        smin = 0
        while i < la and j < lb:
            if sorted_a[i] < sorted_b[j]:
                smin += sorted_a[i] * (lb - j)
                i += 1
            else:
                smin += sorted_b[j] * (la - i)
                j += 1
        return smin
    
    out = []
    # For each query, we:
    #  1) Take the first x elements of A, sort them
    #  2) Take the first y elements of B, sort them
    #  3) sumMin = sum_of_min(...)
    #  4) answer = y * sum(A[:x]) + x * sum(B[:y]) - 2 * sumMin
    #    = y*A_sum[x] + x*B_sum[y] - 2*sumMin
    
    # Note: This is a direct, correct approach, but can be expensive for large x,y.
    # It suffices for correctness and matches the problem statement.
    
    for (x, y) in queries:
        # Sort the first x of A and first y of B
        a_part = sorted(A[:x])
        b_part = sorted(B[:y])
        s_min = sum_of_min(a_part, b_part)
        
        # sum_of_abs = y * (sum of A[:x]) + x * (sum of B[:y]) - 2 * s_min
        ans = y * A_sum[x] + x * B_sum[y] - 2 * s_min
        out.append(str(ans))
    
    print("
".join(out))

# Don't forget to call main()
if __name__ == "__main__":
    main()