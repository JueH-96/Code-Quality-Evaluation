def solve():
    N, Q = map(int, input().split())
    S = list(map(int, input()))
    queries = [list(map(int, input().split())) for _ in range(Q)]

    # Flip the bits in the range [L, R]
    def flip(L, R):
        for i in range(L-1, R):
            S[i] ^= 1

    # Check if the substring [L, R] is a good string
    def is_good(L, R):
        for i in range(L, R):
            if S[i] == S[i+1]:
                return False
        return True

    # Process the queries
    for query in queries:
        if query[0] == 1:
            flip(query[1], query[2])
        else:
            print('Yes' if is_good(query[1]-1, query[2]-1) else 'No')

solve()