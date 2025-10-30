import sys
import bisect

def read_tokens():
    data = sys.stdin.read().split()
    for token in data:
        yield token

def main():
    reader = read_tokens()
    N = int(next(reader))
    X = [int(next(reader)) for _ in range(N)]
    P = [int(next(reader)) for _ in range(N)]
    Q = int(next(reader))
    queries = [(int(next(reader)), int(next(reader))) for _ in range(Q)]
    
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i + 1] = prefix[i] + P[i]
    
    for L, R in queries:
        left_index = bisect.bisect_left(X, L)
        right_index = bisect.bisect_right(X, R)
        answer = prefix[right_index] - prefix[left_index]
        print(answer)

if __name__ == "__main__":
    main()