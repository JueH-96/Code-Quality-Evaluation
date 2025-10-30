import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    ans = 0
    # j_ptr is the pointer for the potential bottom mochi A[j_ptr]
    # i is the pointer for the top mochi A[i]
    j_ptr = 0 
    for i in range(N):
        # We are looking for A[j_ptr] such that A[j_ptr] >= 2 * A[i]
        # This is equivalent to A[i] <= A[j_ptr]/2
        
        target_val = 2 * A[i]
        
        # Advance pointer j_ptr to find the first element A[j_ptr] >= target_val.
        # j_ptr only moves forward across all iterations of i.
        while j_ptr < N and A[j_ptr] < target_val:
            j_ptr += 1
        
        # If j_ptr reaches N, it means no mochi from A[j_ptr] onwards is large enough.
        # The number of valid bottom mochi is N - j_ptr. If j_ptr = N, this is 0.
        # As proved in the thought process, any k >= j_ptr implies k > i,
        # so the condition that mochi must be distinct is satisfied.
        ans += (N - j_ptr)
        
    print(ans)

if __name__ == '__main__':
    main()