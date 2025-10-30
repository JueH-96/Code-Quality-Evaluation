# YOUR CODE HERE
import sys, bisect

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    P = []
    R = []
    C = []
    idx = 2
    for _ in range(N):
        T = int(data[idx])
        X = int(data[idx+1])
        if T ==0:
            P.append(X)
        elif T ==1:
            R.append(X)
        else:
            C.append(X)
        idx +=2
    P.sort(reverse=True)
    R.sort(reverse=True)
    C.sort(reverse=True)
    # Compute prefix sums
    P_prefix = [0]
    for x in P:
        P_prefix.append(P_prefix[-1]+x)
    R_prefix = [0]
    for x in R:
        R_prefix.append(R_prefix[-1]+x)
    C_prefix = [0]
    for x in C:
        C_prefix.append(C_prefix[-1]+x)
    len_P = len(P)
    len_R = len(R)
    len_C = len(C)
    # Precompute c_required[k]
    # C_prefix is sorted non-decreasing
    c_required = [0]*(len_R+1)
    for k in range(1, len_R+1):
        c = bisect.bisect_left(C_prefix, k)
        if c > len_C:
            c_required[k] = float('inf')
        else:
            c_required[k] = c
    max_sum = 0
    max_p = min(len_P, M)
    for p in range(0, max_p+1):
        available = M - p
        # Need to find max k such that c_required[k] +k <= available
        # Binary search on k
        left =0
        right = min(len_R, available)
        best_k =0
        while left <= right:
            mid = (left + right)//2
            if c_required[mid] + mid <= available:
                best_k = mid
                left = mid +1
            else:
                right = mid -1
        current_sum = P_prefix[p] + R_prefix[best_k]
        if current_sum > max_sum:
            max_sum = current_sum
    print(max_sum)

if __name__ == "__main__":
    main()