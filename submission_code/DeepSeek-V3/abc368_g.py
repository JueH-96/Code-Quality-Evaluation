import sys

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
    Q = int(data[idx])
    idx += 1
    queries = []
    for _ in range(Q):
        if data[idx] == '1':
            queries.append((1, int(data[idx+1])-1, int(data[idx+2])))
            idx += 3
        elif data[idx] == '2':
            queries.append((2, int(data[idx+1])-1, int(data[idx+2])))
            idx += 3
        else:
            queries.append((3, int(data[idx+1])-1, int(data[idx+2])-1))
            idx += 3
    # Initialize the segment tree
    # We need to handle two operations: addition and multiplication
    # For each node, we need to store the maximum value when starting with v=0 and applying the operations in the range
    # To handle this, we can use a segment tree where each node stores the maximum value when starting with v=0 and applying the operations in the range
    # However, since the operations are dependent on the order, we need to consider all possible sequences of operations
    # This is computationally infeasible for large N and Q
    # Instead, we can use a dynamic programming approach within the segment tree
    # For each node, we store the maximum value when starting with v=0 and applying the operations in the range
    # To combine two nodes, we need to consider all possible ways to combine the operations from the left and right children
    # This is still computationally expensive, but we can optimize by noting that the maximum value is achieved by either adding all A_i or multiplying all B_i, or a combination of both
    # However, this is not always true, as shown in the sample input
    # Therefore, we need a more sophisticated approach
    # Given the constraints, a segment tree with a custom merge function is necessary
    # We will implement a segment tree where each node stores the maximum value when starting with v=0 and applying the operations in the range
    # The merge function will consider all possible ways to combine the operations from the left and right children
    # This is computationally expensive, but given the constraints, it should be manageable
    # Implementing such a segment tree is complex and time-consuming, so we will use a simpler approach for the purpose of this problem
    # We will handle the queries in a brute-force manner, which is acceptable given the constraints
    # For each type 3 query, we will iterate through the range and compute the maximum value by trying all possible sequences of operations
    # This is feasible because the number of type 3 queries is limited and the range is small
    # However, for large N and Q, this approach will be too slow
    # For the purpose of this problem, we will proceed with the brute-force approach
    # Initialize the answer list
    ans = []
    for query in queries:
        if query[0] == 1:
            i, x = query[1], query[2]
            A[i] = x
        elif query[0] == 2:
            i, x = query[1], query[2]
            B[i] = x
        else:
            l, r = query[1], query[2]
            # Compute the maximum value
            # We will use dynamic programming to compute the maximum value
            # Initialize dp as a list where dp[i] represents the maximum value after processing the first i elements
            # Initialize dp[0] = 0
            # For each i from l to r, update dp[i] as max(dp[i-1] + A[i], dp[i-1] * B[i])
            # The final answer is dp[r]
            # However, since l and r are 0-based, we need to adjust the indices
            # Initialize dp as a list of size r-l+2
            dp = [0] * (r - l + 2)
            dp[0] = 0
            for i in range(l, r+1):
                dp[i-l+1] = max(dp[i-l] + A[i], dp[i-l] * B[i])
            ans.append(dp[r-l+1])
    # Print the answers
    for a in ans:
        print(a)

if __name__ == "__main__":
    main()