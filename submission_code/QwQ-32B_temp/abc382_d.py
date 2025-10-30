import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    result = []

    def backtrack(seq):
        k = len(seq)
        if k == N:
            result.append(seq.copy())
            return
        if k == 0:
            start = 1
            max_val = M - 10 * (N - 1)
        else:
            prev = seq[-1]
            start = prev + 10
            remaining = N - (k + 1)
            max_val = M - 10 * remaining
        if start > max_val:
            return
        for next_val in range(start, max_val + 1):
            seq.append(next_val)
            backtrack(seq)
            seq.pop()

    backtrack([])
    print(len(result))
    for seq in result:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()