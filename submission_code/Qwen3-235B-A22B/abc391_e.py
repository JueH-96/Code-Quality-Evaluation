import sys

def main():
    N = int(sys.stdin.readline().strip())
    A = sys.stdin.readline().strip()
    
    # Precompute the original values of each node
    values = [[] for _ in range(N+1)]
    values[N] = [int(c) for c in A]
    
    # Compute values bottom-up
    for l in range(N-1, -1, -1):
        curr = []
        num_nodes = 3 ** l
        for i in range(num_nodes):
            a = values[l+1][3*i]
            b = values[l+1][3*i + 1]
            c = values[l+1][3*i + 2]
            sum_ = a + b + c
            majority = 1 if sum_ >= 2 else 0
            curr.append(majority)
        values[l] = curr
    
    # Compute cost0 and cost1 bottom-up
    cost0 = [[] for _ in range(N+1)]
    cost1 = [[] for _ in range(N+1)]
    
    # Initialize leaves
    leaf_level = N
    leaf_count = len(A)
    cost0[leaf_level] = [0] * leaf_count
    cost1[leaf_level] = [0] * leaf_count
    for i in range(leaf_count):
        if A[i] == '0':
            cost0[leaf_level][i] = 0
            cost1[leaf_level][i] = 1
        else:
            cost0[leaf_level][i] = 1
            cost1[leaf_level][i] = 0
    
    # Process internal nodes
    for l in range(N-1, -1, -1):
        curr_cost0 = []
        curr_cost1 = []
        num_nodes = 3 ** l
        for i in range(num_nodes):
            child0 = 3 * i
            child1 = 3 * i + 1
            child2 = 3 * i + 2
            
            # Get child costs
            X_c0 = cost0[l+1][child0]
            Y_c0 = cost0[l+1][child1]
            Z_c0 = cost0[l+1][child2]
            X_c1 = cost1[l+1][child0]
            Y_c1 = cost1[l+1][child1]
            Z_c1 = cost1[l+1][child2]
            
            # Compute cost0
            sum_A = X_c0 + Y_c0 + Z_c0
            sum_B1 = X_c1 + Y_c0 + Z_c0
            sum_B2 = X_c0 + Y_c1 + Z_c0
            sum_B3 = X_c0 + Y_c0 + Z_c1
            sum_B = min(sum_B1, sum_B2, sum_B3)
            node_cost0 = min(sum_A, sum_B)
            
            # Compute cost1
            sum_A_1 = X_c1 + Y_c1 + Z_c1
            sum_B1_1 = X_c0 + Y_c1 + Z_c1
            sum_B2_1 = X_c1 + Y_c0 + Z_c1
            sum_B3_1 = X_c1 + Y_c1 + Z_c0
            sum_B_1 = min(sum_B1_1, sum_B2_1, sum_B3_1)
            node_cost1 = min(sum_A_1, sum_B_1)
            
            curr_cost0.append(node_cost0)
            curr_cost1.append(node_cost1)
        
        cost0[l] = curr_cost0
        cost1[l] = curr_cost1
    
    # Determine the answer based on root's original value
    root_val = values[0][0]
    if root_val == 0:
        ans = cost1[0][0]
    else:
        ans = cost0[0][0]
    
    print(ans)

if __name__ == "__main__":
    main()