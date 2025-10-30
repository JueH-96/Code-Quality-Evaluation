import sys
import threading

def main():
    import sys
    data = sys.stdin.buffer.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # prefix_distinct[i] = number of distinct values in A[0..i]
    prefix_distinct = [0] * N
    seen = [False] * (N + 1)
    cnt = 0
    for i, v in enumerate(A):
        if not seen[v]:
            seen[v] = True
            cnt += 1
        prefix_distinct[i] = cnt

    # suffix_distinct[i] = number of distinct values in A[i..N-1]
    suffix_distinct = [0] * N
    seen2 = [False] * (N + 1)
    cnt2 = 0
    for i in range(N - 1, -1, -1):
        v = A[i]
        if not seen2[v]:
            seen2[v] = True
            cnt2 += 1
        suffix_distinct[i] = cnt2

    # Try all splits between i and i+1, i = 0..N-2
    ans = 0
    for i in range(N - 1):
        s = prefix_distinct[i] + suffix_distinct[i + 1]
        if s > ans:
            ans = s

    print(ans)

if __name__ == "__main__":
    main()