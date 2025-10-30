def main():
    import sys
    sys.setrecursionlimit(1 << 25)

    N, X, Y = map(int, sys.stdin.readline().split())
    dishes = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]

    best = 0
    used = [False] * N

    def backtrack(sweetness, saltiness, count):
        nonlocal best
        if count + (N - sum(used)) <= best:
            return
        if count > best:
            best = count
        for i in range(N):
            if not used[i]:
                new_sweetness = sweetness + dishes[i][0]
                new_saltiness = saltiness + dishes[i][1]
                if new_sweetness <= X and new_saltiness <= Y:
                    used[i] = True
                    backtrack(new_sweetness, new_saltiness, count + 1)
                    used[i] = False

    backtrack(0, 0, 0)
    print(best)

if __name__ == "__main__":
    main()