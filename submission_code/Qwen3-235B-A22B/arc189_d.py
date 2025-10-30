import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    result = [0] * N

    processed = [False] * N

    for i in range(N):
        if processed[i]:
            continue

        L = i
        R = i
        s = A[i]
        # Expand as much as possible
        while True:
            merged = False
            # Expand left
            new_L = L
            while new_L > 0 and A[new_L - 1] < s:
                s += A[new_L - 1]
                new_L -= 1
            # Expand right
            new_R = R
            while new_R < N - 1 and A[new_R + 1] < s:
                s += A[new_R + 1]
                new_R += 1
            if new_L == L and new_R == R:
                break
            L, R = new_L, new_R
            merged = True
            if not merged:
                break

        # Now assign the result for all positions from L to R
        # But we need to compute the actual sum, not s
        # Because the merging steps might have added multiple times
        total = 0
        for k in range(L, R + 1):
            total += A[k]
        for k in range(L, R + 1):
            result[k] = total
            processed[k] = True

    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()