import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    # Every single element subarray is an AP
    ans = n
    if n >= 2:
        # Compute differences
        # Count runs of equal differences
        run_len = 1
        prev_diff = A[1] - A[0]
        for i in range(2, n):
            d = A[i] - A[i-1]
            if d == prev_diff:
                run_len += 1
            else:
                # Add number of subarrays in this run of length run_len:
                # run_len*(run_len+1)//2 covers all lengths >=1 in D,
                # which correspond to APs of length >=2 in A.
                ans += run_len * (run_len + 1) // 2
                prev_diff = d
                run_len = 1
        # add the last run
        ans += run_len * (run_len + 1) // 2
    print(ans)

if __name__ == "__main__":
    main()