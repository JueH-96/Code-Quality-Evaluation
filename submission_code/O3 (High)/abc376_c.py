import sys

def main() -> None:
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N - 1)]

    A.sort()
    B.sort()

    # prefix_ok[i] : the first i toys (A[0] … A[i-1]) all fit
    #                into the first i boxes (B[0] … B[i-1])
    prefix_ok = [True] * N          # size N, indices 0 … N-1
    ok = True
    for i in range(N - 1):          # we can only compare up to len(B)=N-1
        ok = ok and (A[i] <= B[i])
        prefix_ok[i + 1] = ok       # validity for first i+1 toys

    # suffix_ok[i] : the toys A[i+1] … A[N-1] all fit into
    #                boxes B[i] … B[N-2]
    suffix_ok = [False] * N
    suffix_ok[N - 1] = True         # removing the last toy leaves empty suffix
    ok = True
    for i in range(N - 2, -1, -1):  # i runs N-2 … 0
        ok = ok and (A[i + 1] <= B[i])
        suffix_ok[i] = ok

    answer = None
    for i in range(N):
        if prefix_ok[i] and suffix_ok[i]:
            candidate = A[i]        # toy i goes into the new box
            if answer is None or candidate < answer:
                answer = candidate

    print(answer if answer is not None else -1)

if __name__ == "__main__":
    main()