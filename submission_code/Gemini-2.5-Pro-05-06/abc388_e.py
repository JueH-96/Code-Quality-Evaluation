import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # A is 0-indexed, N is the count of elements.
    # A_i in problem statement (1-indexed) corresponds to A[i-1] in 0-indexed list.
    
    # To make k_val kagamimochi:
    # Top mochi are A[0], ..., A[k_val-1]
    # Bottom mochi are A[N-k_val], ..., A[N-1]
    # We need to check if 2 * A[i] <= A[N-k_val+i] for i from 0 to k_val-1.

    def check(k_val):
        if k_val == 0: # Base case: 0 pairs are always possible.
            return True
        
        # This condition (2 * k_val <= N) is inherently satisfied
        # because k_val in binary search will not exceed N // 2.
        # if 2 * k_val > N:
        #     return False
        
        for i in range(k_val):
            # Top mochi is A[i]
            # Bottom mochi is A[N - k_val + i]
            top_mochi_val = A[i]
            bottom_mochi_val = A[N - k_val + i]
            
            # Condition: top_mochi_val <= bottom_mochi_val / 2
            # which is equivalent to 2 * top_mochi_val <= bottom_mochi_val
            if 2 * top_mochi_val > bottom_mochi_val:
                return False # This specific pair is not valid
        
        return True # All k_val pairs are valid

    # Binary search for the maximum K
    # K can range from 0 to N // 2
    low = 0
    high = N // 2 
    # max_k_found will store the maximum K found so far that satisfies `check`.
    # It's initialized to 0 because check(0) is always true.
    max_k_found = 0 

    while low <= high:
        mid = low + (high - low) // 2
        
        if check(mid):
            max_k_found = mid   # mid is a possible value for K, try for a larger K
            low = mid + 1       # Search in the right half: [mid+1, high]
        else:
            high = mid - 1      # mid is too large, try smaller K. Search in left half: [low, mid-1]
            
    print(max_k_found)

if __name__ == '__main__':
    solve()