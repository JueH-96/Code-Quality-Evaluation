import sys

def main() -> None:
    input = sys.stdin.readline

    N = int(input())
    L = []
    R = []
    sum_L = 0
    sum_R = 0

    for _ in range(N):
        l, r = map(int, input().split())
        L.append(l)
        R.append(r)
        sum_L += l
        sum_R += r

    # Check if 0 lies inside the achievable sum interval [sum_L, sum_R]
    if sum_L > 0 or sum_R < 0:
        print("No")
        return

    # Construct one concrete solution
    need = -sum_L          # how much we still have to add to reach 0
    X = []
    for l, r in zip(L, R):
        add = min(need, r - l)
        X.append(l + add)
        need -= add

    # `need` must be 0 now
    print("Yes")
    print(" ".join(map(str, X)))

if __name__ == "__main__":
    main()