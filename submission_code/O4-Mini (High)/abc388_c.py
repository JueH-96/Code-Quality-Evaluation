import sys

def main():
    input = sys.stdin.readline
    # Read N
    line = input().split()
    if not line:
        return
    N = int(line[0])
    # Read list A
    A = list(map(int, input().split()))
    
    res = 0
    k = 0
    # Two‐pointer: for each i as bottom, advance k to count how many j<i satisfy A[j]*2 <= A[i]
    for i in range(N):
        # A is sorted ascending, so once A[k]*2 > A[i], no further j will satisfy
        while k < i and A[k] * 2 <= A[i]:
            k += 1
        # All indices [0..k-1] are valid tops for bottom i
        res += k

    # Output the total count
    print(res)

# Call main
main()