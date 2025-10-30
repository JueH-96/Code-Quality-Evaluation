import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    A = list(map(int, data[idx:idx+N]))
    idx += N
    B = list(map(int, data[idx:idx+N]))
    idx += N
    K = int(data[idx])
    idx += 1
    queries = []
    for _ in range(K):
        X = int(data[idx])
        Y = int(data[idx+1])
        queries.append((X, Y))
        idx += 2
    
    # Precompute prefix sums for A and B
    A_sorted = sorted(A)
    B_sorted = sorted(B)
    
    # Precompute prefix sums for A_sorted and B_sorted
    prefix_A = [0] * (N + 1)
    for i in range(N):
        prefix_A[i+1] = prefix_A[i] + A_sorted[i]
    
    prefix_B = [0] * (N + 1)
    for i in range(N):
        prefix_B[i+1] = prefix_B[i] + B_sorted[i]
    
    for x, y in queries:
        # Get the first x elements of A and first y elements of B
        A_sub = A_sorted[:x]
        B_sub = B_sorted[:y]
        
        # To compute the sum of |a_i - b_j| for all i in 1..x and j in 1..y
        # We can use the fact that |a_i - b_j| = a_i - b_j if a_i >= b_j, else b_j - a_i
        # So, for each a_i, we can find the number of b_j <= a_i and the number of b_j > a_i
        # Then, sum_{j=1}^y |a_i - b_j| = a_i * count_leq - sum_{b_j <= a_i} b_j + sum_{b_j > a_i} b_j - a_i * count_gt
        # Which simplifies to a_i * (count_leq - count_gt) - (sum_{b_j <= a_i} b_j - sum_{b_j > a_i} b_j)
        # But since count_leq + count_gt = y, count_leq - count_gt = 2 * count_leq - y
        # So, sum_{j=1}^y |a_i - b_j| = a_i * (2 * count_leq - y) - (sum_{b_j <= a_i} b_j - sum_{b_j > a_i} b_j)
        # But sum_{b_j <= a_i} b_j + sum_{b_j > a_i} b_j = total_sum_B
        # So, sum_{b_j <= a_i} b_j - sum_{b_j > a_i} b_j = 2 * sum_{b_j <= a_i} b_j - total_sum_B
        # So, sum_{j=1}^y |a_i - b_j| = a_i * (2 * count_leq - y) - (2 * sum_{b_j <= a_i} b_j - total_sum_B)
        # Which simplifies to a_i * (2 * count_leq - y) - 2 * sum_{b_j <= a_i} b_j + total_sum_B
        
        total_sum_B = prefix_B[y]
        total_sum_A = prefix_A[x]
        
        res = 0
        for a in A_sub:
            # Find the number of b_j <= a in B_sub
            count_leq = bisect.bisect_right(B_sub, a)
            sum_leq = prefix_B[count_leq]
            # sum_gt = total_sum_B - sum_leq
            # sum_leq - sum_gt = 2 * sum_leq - total_sum_B
            # a * (2 * count_leq - y) - (2 * sum_leq - total_sum_B)
            term = a * (2 * count_leq - y) - (2 * sum_leq - total_sum_B)
            res += term
        
        print(res)

if __name__ == "__main__":
    main()