import sys

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    A.sort()
    ans = 0
    j = 0
    for i in range(N):
        # extend j as far as A[j] < A[i] + M
        limit = A[i] + M
        while j < N and A[j] < limit:
            j += 1
        # now [i, j) is the maximal window starting at i
        curr = j - i
        if curr > ans:
            ans = curr
    sys.stdout.write(str(ans))

main()