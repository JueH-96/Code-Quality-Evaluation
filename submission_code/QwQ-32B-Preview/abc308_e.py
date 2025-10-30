import sys
from bisect import bisect_left

def main():
    import sys
    from bisect import bisect_left
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    S = data[N+1]
    
    # Separate positions based on S
    M_positions = []
    E_positions = []
    X_positions = []
    for idx, char in enumerate(S):
        pos = idx + 1  # 1-based index
        val = A[idx]
        if char == 'M':
            M_positions.append(pos)
        elif char == 'E':
            E_positions.append((pos, val))
        elif char == 'X':
            X_positions.append(pos)
    
    # Further separate M and X positions based on A values
    M_positions_A0 = [pos for pos in M_positions if A[pos-1] == 0]
    M_positions_A1 = [pos for pos in M_positions if A[pos-1] == 1]
    M_positions_A2 = [pos for pos in M_positions if A[pos-1] == 2]
    
    X_positions_A0 = [pos for pos in X_positions if A[pos-1] == 0]
    X_positions_A1 = [pos for pos in X_positions if A[pos-1] == 1]
    X_positions_A2 = [pos for pos in X_positions if A[pos-1] == 2]
    
    # Sort these lists for binary search
    M_positions_A0.sort()
    M_positions_A1.sort()
    M_positions_A2.sort()
    X_positions_A0.sort()
    X_positions_A1.sort()
    X_positions_A2.sort()
    
    # Precompute mex values for all possible combinations
    mex_values = {}
    for a in range(3):
        for b in range(3):
            for c in range(3):
                s = {a, b, c}
                for i in range(4):
                    if i not in s:
                        mex_values[(a, b, c)] = i
                        break
    
    total_sum = 0
    for j_pos, A_j in E_positions:
        # Count M positions before j
        count_M_A0 = bisect_left(M_positions_A0, j_pos)
        count_M_A1 = bisect_left(M_positions_A1, j_pos)
        count_M_A2 = bisect_left(M_positions_A2, j_pos)
        
        # Count X positions after j
        count_X_A0 = len(X_positions_A0) - bisect_left(X_positions_A0, j_pos)
        count_X_A1 = len(X_positions_A1) - bisect_left(X_positions_A1, j_pos)
        count_X_A2 = len(X_positions_A2) - bisect_left(X_positions_A2, j_pos)
        
        # Compute contributions to the sum
        # for each combination of A_i and A_k
        for A_i in range(3):
            for A_k in range(3):
                mex_val = mex_values[(A_i, A_j, A_k)]
                if A_i == 0:
                    count_M = count_M_A0
                elif A_i == 1:
                    count_M = count_M_A1
                else:
                    count_M = count_M_A2
                if A_k == 0:
                    count_X = count_X_A0
                elif A_k == 1:
                    count_X = count_X_A1
                else:
                    count_X = count_X_A2
                total_sum += count_M * count_X * mex_val
    
    print(total_sum)

if __name__ == "__main__":
    main()