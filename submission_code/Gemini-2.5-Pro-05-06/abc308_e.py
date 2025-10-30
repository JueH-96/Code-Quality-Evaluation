import sys

def solve():
    # Precompute mex_table[vM][Aj_val][vX]
    mex_table = [[[0]*3 for _ in range(3)] for _ in range(3)]
    for val1 in range(3):
        for val2 in range(3):
            for val3 in range(3):
                s = {val1, val2, val3}
                current_mex = 0
                if 0 not in s:
                    current_mex = 0
                elif 1 not in s:
                    current_mex = 1
                elif 2 not in s:
                    current_mex = 2
                else: # 0, 1, 2 are all in s
                    current_mex = 3
                mex_table[val1][val2][val3] = current_mex

    N = int(sys.stdin.readline())
    A_str = sys.stdin.readline().split()
    A = [int(x) for x in A_str]
    S = sys.stdin.readline().strip()

    # suffix_countX[p][val] = count of k >= p such that S[k]=='X' and A[k]==val
    # N+1 elements to handle suffix_countX[N] as all zeros (base case for k > N-1)
    suffix_countX = [[0, 0, 0] for _ in range(N + 1)]
    for p in range(N - 1, -1, -1):
        # Copy counts from p+1
        for val_type in range(3):
            suffix_countX[p][val_type] = suffix_countX[p + 1][val_type]
        
        if S[p] == 'X':
            # A[p] is guaranteed to be 0, 1, or 2
            suffix_countX[p][A[p]] += 1

    total_sum = 0
    # current_countM[val] = count of i < j such that S[i]=='M' and A[i]==val
    current_countM = [0, 0, 0]

    for j in range(N): # Iterate through all possible middle indices j
        if S[j] == 'E':
            Aj_val = A[j] # A[j] is guaranteed to be 0, 1, or 2
            for vM in range(3): # Iterate over possible values for A[i]
                num_M = current_countM[vM]
                if num_M == 0: # No 'M' with A[i]=vM found yet for i < j
                    continue
                
                for vX in range(3): # Iterate over possible values for A[k]
                    # Counts for k > j are from suffix_countX[j+1]
                    # If j=N-1, j+1=N, suffix_countX[N] is [0,0,0], so num_X will be 0.
                    num_X = suffix_countX[j + 1][vX]
                    if num_X == 0: # No 'X' with A[k]=vX found for k > j
                        continue
                    
                    mval = mex_table[vM][Aj_val][vX]
                    total_sum += num_M * num_X * mval
        
        elif S[j] == 'M':
            # This index j could be an 'M' for a future 'E'
            # A[j] is guaranteed to be 0, 1, or 2
            current_countM[A[j]] += 1
                 
    sys.stdout.write(str(total_sum) + "
")

if __name__ == '__main__':
    solve()