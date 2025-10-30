import sys

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    max_a = max(A)
    freq = [0] * (max_a + 1)
    for num in A:
        freq[num] += 1

    cnt = [0] * (max_a + 1)
    for d in range(1, max_a + 1):
        for m in range(d, max_a + 1, d):
            cnt[d] += freq[m]

    pos = [[] for _ in range(max_a + 1)]
    for i in range(N):
        num = A[i]
        pos[num].append(i)

    ans = [0] * N

    for d in range(max_a, 0, -1):
        if cnt[d] >= K:
            m = d
            while m <= max_a:
                for idx in pos[m]:
                    if ans[idx] == 0:
                        ans[idx] = d
                m += d

    for a in ans:
        print(a)

if __name__ == "__main__":
    main()