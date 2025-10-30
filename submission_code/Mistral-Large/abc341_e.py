import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0

    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    S = data[index]
    index += 1

    queries = []
    for _ in range(Q):
        query_type = int(data[index])
        L = int(data[index + 1])
        R = int(data[index + 2])
        index += 3
        queries.append((query_type, L, R))

    S = list(S)
    flipped = [False] * (N + 1)

    results = []

    for query_type, L, R in queries:
        if query_type == 1:
            flipped[L-1] ^= True
            if R < N:
                flipped[R] ^= True
        else:
            is_good = True
            for i in range(L, R + 1):
                actual_char = '0' if S[i-1] == '1' else '1' if flipped[i-1] else S[i-1]
                if i < R + 1 and actual_char == (S[i] if not flipped[i] else '0' if S[i] == '1' else '1'):
                    is_good = False
                    break
            results.append("Yes" if is_good else "No")

    sys.stdout.write("
".join(results) + "
")

if __name__ == "__main__":
    main()