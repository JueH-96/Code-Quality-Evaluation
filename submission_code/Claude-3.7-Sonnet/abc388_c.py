import bisect

def count_kagamimochi(N, A):
    count = 0
    for i in range(N):
        # Find the smallest index j such that A[j] >= 2 * A[i]
        j = bisect.bisect_left(A, 2 * A[i])
        # There are N - j mochi that satisfy A[j] >= 2 * A[i]
        if i >= j:
            # If i >= j, we need to exclude the mochi i itself
            count += N - j - 1
        else:
            count += N - j
    
    return count

N = int(input())
A = list(map(int, input().split()))
print(count_kagamimochi(N, A))