import sys
from collections import deque

def check(k, N, A):
    """
    Checks if a Pyramid Sequence of size k can be formed from sequence A.
    """
    L_k = 2 * k - 1  # Length of Pyramid Sequence of size k

    # If the target pyramid is longer than the available sequence, it's impossible.
    if L_k > N:
        return False

    # Precompute B[i] = A[i] - i and C[i] = A[i] + i
    # These transformations help simplify the conditions for pyramid formation.
    # Condition 1: A[idx] >= (idx-l)+1  =>  A[idx] - idx >= 1 - l  =>  B[idx] >= 1 - l
    # Condition 2: A[idx] >= (2k-1)-(idx-l) => A[idx] + idx >= l + (2k-1) => C[idx] >= l + L_k
    
    B = [0] * N
    C = [0] * N
    for i in range(N):
        B[i] = A[i] - i
        C[i] = A[i] + i

    dqB = deque() # Deque for sliding window minimum of B values
    dqC = deque() # Deque for sliding window minimum of C values

    # Initialize deques for the first possible window (l=0)
    # B window covers indices [0, k-1]
    for i in range(k):
        while dqB and B[dqB[-1]] >= B[i]:
            dqB.pop()
        dqB.append(i)

    # C window covers indices [k, 2k-2]
    # If k=1, this range is empty ([1, 0]), so dqC remains empty.
    if k > 1: 
        for i in range(k, L_k): # Iterate from k to (2k-1)-1
            while dqC and C[dqC[-1]] >= C[i]:
                dqC.pop()
            dqC.append(i)

    # Iterate through all possible starting positions 'l' for the pyramid in A
    # 'l' ranges from 0 up to N - L_k
    for l in range(N - L_k + 1):
        # 1. Get current minimums for the windows corresponding to 'l'
        minB_val = B[dqB[0]]
        minC_val = float('inf') # Default to infinity, meaning condition is trivially met if C window is empty (k=1)
        if k > 1:
            minC_val = C[dqC[0]]

        # 2. Check if both conditions are met for the current 'l'
        if minB_val >= (1 - l) and minC_val >= (l + L_k):
            return True # Found a valid subsegment for this k

        # 3. Update deques for the next iteration (l+1)
        # This step is only necessary if there are more 'l' values to check
        if l < N - L_k:
            # Remove elements that fall out of the window from the left
            # B window shifts from [l, l+k-1] to [l+1, l+k]
            if dqB and dqB[0] == l:
                dqB.popleft()

            # C window shifts from [l+k, l+2k-2] to [l+k+1, l+2k-1]
            if k > 1 and dqC and dqC[0] == l + k:
                dqC.popleft()

            # Add new elements to the right of the window
            # New rightmost element for B window is at index l+k
            next_B_idx_to_add = l + k
            while dqB and B[dqB[-1]] >= B[next_B_idx_to_add]:
                dqB.pop()
            dqB.append(next_B_idx_to_add)

            # New rightmost element for C window is at index l+(2k-1)
            next_C_idx_to_add = l + L_k 
            if k > 1: # C window exists
                while dqC and C[dqC[-1]] >= C[next_C_idx_to_add]:
                    dqC.pop()
                dqC.append(next_C_idx_to_add)
    
    return False # No valid subsegment found for this k

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Binary search for the maximum possible k
    low = 1  # Minimum possible k (Pyramid Sequence (1))
    high = (N + 1) // 2 # Maximum possible k given length constraint (2k-1 <= N)
    ans = 1 # Initialize answer; k=1 is always possible

    while low <= high:
        mid = (low + high) // 2
        if check(mid, N, A):
            ans = mid      # mid is possible, try for a larger k
            low = mid + 1
        else:
            high = mid - 1 # mid is not possible, try for a smaller k
    
    print(ans)

if __name__ == '__main__':
    main()