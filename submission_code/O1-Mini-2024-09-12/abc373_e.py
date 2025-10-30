# YOUR CODE HERE
import sys, bisect
def main():
    import sys
    import math
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:3+N]))
    sumA = sum(A)
    rem = K - sumA
    A_sorted = sorted(A)
    results = []
    for i in range(N):
        A_i = A[i]
        low = 0
        high = rem
        ans = -1
        while low <= high:
            mid = (low + high) //2
            t = A_i + 2*mid - rem
            idx = bisect.bisect_right(A_sorted, t)
            cnt = N - idx
            if A_i > t:
                cnt -=1
            if cnt < M:
                ans = mid
                high = mid -1
            else:
                low = mid +1
        if ans == -1:
            results.append(-1)
        else:
            if ans <=0:
                # Check if X=0 satisfies
                t0 = A_i - rem
                idx0 = bisect.bisect_right(A_sorted, t0)
                cnt0 = N - idx0
                if A_i > t0:
                    cnt0 -=1
                if cnt0 < M:
                    results.append(0)
                else:
                    results.append(ans)
            else:
                results.append(ans)
    print(' '.join(map(str, results)))
if __name__ == "__main__":
    main()