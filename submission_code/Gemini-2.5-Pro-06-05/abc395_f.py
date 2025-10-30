import sys

# For faster I/O in Python
input = sys.stdin.readline

def solve():
    """
    Main function to orchestrate the solution.
    """
    # Read problem parameters
    try:
        line = input()
        if not line: return
        N, X = map(int, line.split())
    except (IOError, ValueError):
        return

    # Read tooth data and pre-calculate sums
    U = [0] * N
    D = [0] * N
    total_S = 0
    max_S = 0
    for i in range(N):
        u_val, d_val = map(int, input().split())
        U[i] = u_val
        D[i] = d_val
        s_val = u_val + d_val
        total_S += s_val
        max_S = max(max_S, s_val)

    def check(H):
        """
        Check if a valid tooth configuration exists for a given sum H.
        A configuration is valid if we can find new upper tooth lengths U'_i such that:
        1. max(0, H - D_i) <= U'_i <= min(U_i, H)
        2. |U'_i - U'_{i+1}| <= X
        This is determined by propagating constraints on the possible range for each U'_i.
        """
        # Initial valid ranges for each U'_i based on H
        L = [0] * N
        R = [0] * N
        for i in range(N):
            l_val = max(0, H - D[i])
            r_val = min(U[i], H)
            
            if l_val > r_val:
                return False
            L[i] = l_val
            R[i] = r_val

        # a, b will store the dynamically tightened ranges for U'_i
        a = [0] * N
        b = [0] * N

        # Forward pass: propagate constraints from i-1 to i
        a[0] = L[0]
        b[0] = R[0]
        for i in range(1, N):
            a[i] = max(L[i], a[i-1] - X)
            b[i] = min(R[i], b[i-1] + X)
            if a[i] > b[i]:
                return False

        # Backward pass: propagate constraints from i+1 to i
        for i in range(N - 2, -1, -1):
            a[i] = max(a[i], a[i+1] - X)
            b[i] = min(b[i], b[i+1] + X)
            if a[i] > b[i]:
                return False
        
        return True

    # Binary search for the maximum possible H
    low = 0
    high = max_S
    ans_H = 0

    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            ans_H = mid
            low = mid + 1
        else:
            high = mid - 1
            
    # Minimum cost is achieved with the maximum possible H
    min_cost = total_S - N * ans_H
    print(min_cost)

solve()