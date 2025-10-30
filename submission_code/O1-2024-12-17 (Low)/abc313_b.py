def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    edges = list(map(int, input_data[2:]))

    # strong[a][b] = True means "a is stronger than b"
    strong = [[False]*N for _ in range(N)]

    # Read the M pieces of information
    idx = 0
    for _ in range(M):
        A = edges[idx] - 1
        B = edges[idx+1] - 1
        idx += 2
        strong[A][B] = True

    # Compute transitive closure using Floyd-Warshall style
    for k in range(N):
        for i in range(N):
            if strong[i][k]:
                for j in range(N):
                    strong[i][j] = strong[i][j] or strong[k][j]

    # Find all candidates who have no one stronger
    # i.e. no j such that strong[j][i] is True
    candidates = []
    for i in range(N):
        # check if there's some j != i with j->i
        can_be_top = True
        for j in range(N):
            if i != j and strong[j][i]:
                can_be_top = False
                break
        if can_be_top:
            candidates.append(i+1)

    # If exactly one candidate, print it; else print -1
    if len(candidates) == 1:
        print(candidates[0])
    else:
        print(-1)

# Do not forget to call main!
main()