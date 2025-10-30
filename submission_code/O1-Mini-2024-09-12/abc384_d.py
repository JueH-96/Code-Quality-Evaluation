# YOUR CODE HERE
import sys
import bisect

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N_S = sys.stdin.readline().split()
    while len(N_S) <2:
        N_S += sys.stdin.readline().split()
    N = int(N_S[0])
    S = int(N_S[1])
    A = []
    while len(A) < N:
        A += list(map(int, sys.stdin.readline().split()))
    sum_A = sum(A)
    B = A + A
    window_sums = set()
    left =0
    sum_ =0
    for right in range(len(B)):
        sum_ += B[right]
        while sum_ > sum_A and left <= right:
            sum_ -= B[left]
            left +=1
        if sum_ >0:
            window_sums.add(sum_)
    window_sums = sorted(window_sums)
    if sum_A ==0:
        # All window_sums are same, but since A_i >=1, sum_A >=1
        print("No")
        sys.exit()
    k_max = S // sum_A
    for k in range(max(0, k_max -2), k_max +3):
        if k <0:
            continue
        partial_sum = S - k * sum_A
        if partial_sum <=0:
            continue
        idx = bisect.bisect_left(window_sums, partial_sum)
        if idx < len(window_sums) and window_sums[idx] == partial_sum:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()