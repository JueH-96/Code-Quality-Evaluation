import sys
from collections import deque

def solve():
    """
    Solves a single test case of the problem.
    """
    try:
        line = sys.stdin.readline()
        if not line: return False
        H, W = map(int, line.split())
        S = [sys.stdin.readline().strip() for _ in range(H)]
    except (IOError, ValueError):
        return False

    MOD = 998244353

    A = [[(1 if c == 'A' else 0) for c in row] for row in S]

    # Step 1: Feasibility Check
    for i in range(H):
        if sum(A[i]) % 2 != 0:
            print(0)
            return True
    for j in range(W):
        col_sum = sum(A[i][j] for i in range(H))
        if col_sum % 2 != 0:
            print(0)
            return True

    # Step 2: System of Equations for 'B' tiles
    adj = [[] for _ in range(H + W)]

    # Precompute prefix XOR sums for A_matrix
    row_pref_xor = [[0] * (W + 1) for _ in range(H)]
    for i in range(H):
        for j in range(W):
            row_pref_xor[i][j+1] = row_pref_xor[i][j] ^ A[i][j]

    col_pref_xor = [[0] * W for _ in range(H + 1)]
    for j in range(W):
        for i in range(H):
            col_pref_xor[i+1][j] = col_pref_xor[i][j] ^ A[i][j]
            
    # Add edges for 'B' tiles based on equations u_i ^ v_j = C_ij
    for i in range(H):
        for j in range(W):
            if S[i][j] == 'B':
                # Deriving C_ij for u_i ^ v_j = C_ij
                # x_{i,j} = u_i ^ XOR_sum(A[i][k] for k=1..j)
                # y_{i,j} = v_j ^ XOR_sum(A[k][j] for k=1..i)
                # x_{i,j} ^ y_{i,j} = 1
                
                P_ij = row_pref_xor[i][j+1] ^ A[i][0]
                Q_ij = col_pref_xor[i+1][j] ^ A[0][j]
                C = 1 ^ P_ij ^ Q_ij
                
                u_node = i
                v_node = H + j
                adj[u_node].append((v_node, C))
                adj[v_node].append((u_node, C))

    # Step 3: Solve the system using BFS to find components and check consistency
    values = [-1] * (H + W)
    num_components = 0
    is_consistent = True

    for k in range(H + W):
        if values[k] == -1:
            num_components += 1
            values[k] = 0
            q = deque([(k, 0)])
            
            component_consistent = True
            while q:
                p, p_val = q.popleft()
                
                for neighbor, C in adj[p]:
                    neighbor_val = C ^ p_val
                    if values[neighbor] == -1:
                        values[neighbor] = neighbor_val
                        q.append((neighbor, neighbor_val))
                    elif values[neighbor] != neighbor_val:
                        component_consistent = False
                        break
                if not component_consistent:
                    break
            
            if not component_consistent:
                is_consistent = False
                break
    
    if not is_consistent:
        print(0)
    else:
        ans = pow(2, num_components, MOD)
        print(ans)
    
    return True

def main():
    """
    Main function to handle multiple test cases.
    """
    sys.setrecursionlimit(2 * 10**6)
    try:
        T_str = sys.stdin.readline()
        if not T_str: return
        T = int(T_str)
        for _ in range(T):
            if not solve():
                break
    except (IOError, ValueError):
        return

if __name__ == "__main__":
    main()