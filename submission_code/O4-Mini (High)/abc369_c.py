import sys

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # Every single element subarray is an AP
    ans = n
    
    # Count subarrays of length >= 2 with constant adjacent difference
    if n >= 2:
        prev_diff = A[1] - A[0]
        run_len = 1
        # We track runs in the difference array of length (n-1).
        # A run of length L in the difference array contributes L*(L+1)/2 subarrays of A of length>=2.
        for i in range(2, n):
            d = A[i] - A[i-1]
            if d == prev_diff:
                run_len += 1
            else:
                ans += run_len * (run_len + 1) // 2
                prev_diff = d
                run_len = 1
        ans += run_len * (run_len + 1) // 2
    
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()