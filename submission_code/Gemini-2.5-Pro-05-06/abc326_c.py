import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    A.sort()

    max_gifts = 0
    
    # right_idx is a pointer to the element just beyond the current window.
    # Specifically, for a window starting with A[left_idx], A[right_idx-1] is the last element
    # in the window (i.e., A[right_idx-1] < A[left_idx] + M), and A[right_idx] is the first
    # element not in the window (i.e., A[right_idx] >= A[left_idx] + M), 
    # or right_idx == N if all remaining elements are in the window.
    right_idx = 0
    
    for left_idx in range(N):
        # The interval starts effectively at A[left_idx].
        # We are looking for gifts A_k such that A[left_idx] <= A_k < A[left_idx] + M.
        
        limit = A[left_idx] + M
        
        # Advance right_idx while elements are within the current interval's right bound.
        # A[right_idx] must be < limit.
        # right_idx only increases or stays the same, ensuring O(N) overall for this two-pointer part.
        while right_idx < N and A[right_idx] < limit:
            right_idx += 1
        
        # The gifts included are A[left_idx], ..., A[right_idx-1].
        # The number of such gifts is (right_idx - 1) - left_idx + 1 = right_idx - left_idx.
        current_gifts = right_idx - left_idx
        
        if current_gifts > max_gifts:
            max_gifts = current_gifts
            
    print(max_gifts)

if __name__ == '__main__':
    main()