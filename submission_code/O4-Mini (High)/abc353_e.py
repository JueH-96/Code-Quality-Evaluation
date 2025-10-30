import sys
def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    # Read N strings
    S = [next(it) for _ in range(N)]
    # Sort strings lexicographically
    S.sort()
    # Build array of LCPs between adjacent sorted strings
    M = N - 1
    A = [0] * M
    for i in range(M):
        s1 = S[i]
        s2 = S[i+1]
        # compute LCP length
        # iterate up to min-length
        l1 = len(s1)
        l2 = len(s2)
        lm = l1 if l1 < l2 else l2
        cnt = 0
        # compare characters
        # localize for speed
        while cnt < lm and s1[cnt] == s2[cnt]:
            cnt += 1
        A[i] = cnt

    # Sum of all-pairs LCPs is sum of subarray-minimums of A.
    # We compute for each A[i] how many subarrays have A[i] as their minimum,
    # using a monotonic stack approach.

    # left[i] = distance to previous strictly smaller element
    left = [0] * M
    stack = []
    for i in range(M):
        ai = A[i]
        # pop until we find strictly smaller
        while stack and A[stack[-1]] >= ai:
            stack.pop()
        prev = stack[-1] if stack else -1
        left[i] = i - prev
        stack.append(i)

    # right[i] = distance to next smaller-or-equal element
    right = [0] * M
    stack.clear()
    for i in range(M-1, -1, -1):
        ai = A[i]
        # pop until we find <=
        while stack and A[stack[-1]] > ai:
            stack.pop()
        nxt = stack[-1] if stack else M
        right[i] = nxt - i
        stack.append(i)

    # accumulate the total
    total = 0
    for i in range(M):
        # each A[i] contributes A[i] * left[i] * right[i]
        total += A[i] * left[i] * right[i]

    # output the result
    sys.stdout.write(str(total))

if __name__ == '__main__':
    main()