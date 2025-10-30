import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    M = int(data[ptr])
    ptr += 1
    A = [int(data[ptr + j]) for j in range(M)]
    ptr += M
    S = [data[ptr + i] for i in range(N)]
    ptr += N

    for i in range(N):
        current_total_i = sum(A[j] for j in range(M) if S[i][j] == 'o') + (i + 1)
        others = [sum(A[j] for j in range(M) if S[k][j] == 'o') + (k + 1) for k in range(N) if k != i]
        max_other = max(others)
        if current_total_i > max_other:
            print(0)
        else:
            unsolved = [A[j] for j in range(M) if S[i][j] == 'x']
            unsolved_sorted = sorted(unsolved, reverse=True)
            additional_score = 0
            count = 0
            while current_total_i + additional_score <= max_other:
                additional_score += unsolved_sorted[count]
                count += 1
            print(count)

if __name__ == '__main__':
    main()