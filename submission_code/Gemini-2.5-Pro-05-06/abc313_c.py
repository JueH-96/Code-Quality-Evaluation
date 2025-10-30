import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    S = sum(A)
    
    # Target values will be q and q+1
    q = S // N
    # r is the number of elements that should become q+1
    r = S % N 
    # N-r elements should become q

    # Sort A to match smallest elements of A with smallest target values (q)
    # and largest elements of A with largest target values (q+1)
    A.sort()

    # Calculate total absolute difference between current values and target values.
    # The first N-r elements in sorted A are targeted to become q.
    # The remaining r elements in sorted A are targeted to become q+1.
    
    total_abs_diff = 0
    
    for i in range(N):
        target_val = q
        if i >= N - r: # These are the r largest elements, target q+1
            target_val = q + 1
        
        total_abs_diff += abs(A[i] - target_val)
            
    # Each operation (decrease A_x by 1, increase A_y by 1) effectively moves
    # 1 unit of value. If A_x is decreased towards its target and A_y is
    # increased towards its target, this reduces the sum of absolute differences by 2.
    # Therefore, the number of operations is total_abs_diff / 2.
    num_operations = total_abs_diff // 2

    sys.stdout.write(str(num_operations) + "
")

if __name__ == '__main__':
    main()