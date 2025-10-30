import sys

def main():
    import sys
    input = sys.stdin.readline

    N, S = map(int, input().split())
    A = list(map(int, input().split()))

    # Build prefix sums modulo one period:
    # remList[i] = sum of A[0..i-1], for i=0..N-1
    remList = [0] * N
    rem = 0
    for i in range(N):
        remList[i] = rem
        rem += A[i]
    sumA = rem  # total sum of one period

    # 1) First check any non‐wrapping subarray within one block that sums to S
    #    (classic two‐pointer for positive ints)
    curr = 0
    left = 0
    for right in range(N):
        curr += A[right]
        # shrink from left while we exceed S
        while left <= right and curr > S:
            curr -= A[left]
            left += 1
        if curr == S:
            print("Yes")
            return

    # 2) Check for any wrapping/cross‐block subarray.
    #    A necessary and sufficient condition is:
    #      exists i,j in [0..N-1] such that
    #        (remList[j] - remList[i]) + k*sumA = S
    #      for some integer k>=1.
    #    Equivalently: (remList[i] + S) ≡ remList[j] (mod sumA).
    remSet = set(remList)
    for r in remList:
        t = (r + S) % sumA
        if t in remSet:
            print("Yes")
            return

    print("No")


if __name__ == "__main__":
    main()