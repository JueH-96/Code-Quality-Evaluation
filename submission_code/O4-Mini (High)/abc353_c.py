import sys

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0])
    # Read exactly n integers for A
    A = list(map(int, data[1:1+n]))
    A.sort()
    
    M = 10**8
    total_sum = sum(A)
    
    # Count how many pairs (i<j) satisfy A[i] + A[j] >= M
    cnt_exceed = 0
    j = n - 1
    for i, ai in enumerate(A):
        # Decrease j until either j == i or A[i] + A[j] < M
        while j > i and ai + A[j] >= M:
            j -= 1
        # Now for this i, all indices k > max(i, j) have A[i] + A[k] >= M
        # Number of such k is (n-1) - max(i, j)
        # e.g. if j > i, count = (n-1) - j; if j <= i, count = (n-1) - i
        bound = j if j > i else i
        cnt_exceed += (n - 1 - bound)
    
    # Sum over all pairs of (Ai + Aj) = (n-1) * sum(A)
    # For each pair with Ai + Aj >= M, we subtract M once.
    ans = total_sum * (n - 1) - cnt_exceed * M
    
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()