# YOUR CODE HERE
import sys, bisect
def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.read().split()))
    S = sum(A)
    max_possible = S // K
    low = max(A)
    high = S
    def can_partition(x):
        count = 0
        current = 0
        for a in A*2:
            current += a
            if current >= x:
                count +=1
                current =0
                if count >=K:
                    return True
            if count >=K:
                break
        return count >=K
    # Binary search for x
    left = low
    right = S
    while left < right:
        mid = (left + right +1)//2
        if can_partition(mid):
            left = mid
        else:
            right = mid -1
    x = left
    # Now find y
    # We need to find lines where no segment can end at that line with sum >=x
    # Compute prefix sums
    prefix = [0]*(2*N+1)
    for i in range(2*N):
        prefix[i+1] = prefix[i] + A[i%N]
    y =0
    min_prefix = [0]*(2*N+1)
    min_p = prefix[0]
    for i in range(1, 2*N+1):
        min_prefix[i] = min_p
        if prefix[i] < min_p:
            min_p = prefix[i]
    # For each line in first N, check if there exists a j <=i where prefix[i] - prefix[j] >=x and i -j <=N
    possible = [False]*N
    for i in range(1, 2*N+1):
        # The window j should be in [i -N, i)
        j_min = max(0, i -N)
        current_min = min_prefix[i]
        if prefix[i] - current_min >=x:
            if i <=N:
                possible[i-1] = True
            else:
                possible[(i-1)%N] = True
    y = possible.count(False)
    print(x, y)
    
if __name__ == "__main__":
    main()