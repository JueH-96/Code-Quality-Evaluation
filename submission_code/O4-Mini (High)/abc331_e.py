import sys

def main():
    input = sys.stdin.readline
    N, M, L = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Map each main dish to the set of forbidden side dishes
    forbiddens_by_main = {}
    for _ in range(L):
        c, d = map(int, input().split())
        i, j = c - 1, d - 1
        if i in forbiddens_by_main:
            forbiddens_by_main[i].add(j)
        else:
            forbiddens_by_main[i] = {j}

    # Sides sorted by descending price
    b_sorted = sorted(range(M), key=lambda x: B[x], reverse=True)
    # Mains sorted by descending price
    a_sorted = sorted(range(N), key=lambda x: A[x], reverse=True)

    b_max = B[b_sorted[0]]
    answer = 0

    # For each main in descending order, find its best allowed side
    for i in a_sorted:
        # Upper bound check: if even pairing with the most expensive side
        # can't beat current answer, we can stop.
        if A[i] + b_max <= answer:
            break

        forbid_set = forbiddens_by_main.get(i)
        # Scan sides from most expensive down, skip forbidden ones
        for j in b_sorted:
            if forbid_set is not None and j in forbid_set:
                continue
            # Found the best allowed side for main i
            candidate = A[i] + B[j]
            if candidate > answer:
                answer = candidate
            break

    print(answer)

if __name__ == "__main__":
    main()