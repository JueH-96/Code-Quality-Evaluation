import sys

def main():
    n = int(sys.stdin.readline())
    pairs = []
    sum_L = 0
    sum_R = 0
    for _ in range(n):
        L, R = map(int, sys.stdin.readline().split())
        pairs.append((L, R))
        sum_L += L
        sum_R += R

    if sum_L > 0 or sum_R < 0:
        print("No")
        return

    required = -sum_L
    X = [L for L, R in pairs]

    for i in range(n):
        if required <= 0:
            break
        L, R = pairs[i]
        available = R - X[i]
        add = min(available, required)
        X[i] += add
        required -= add

    print("Yes")
    print(' '.join(map(str, X)))

if __name__ == "__main__":
    main()