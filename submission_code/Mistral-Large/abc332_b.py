import sys

def pour_water(K, G, M):
    glass = 0
    mug = 0

    for _ in range(K):
        if glass == G:
            glass = 0
        elif mug == 0:
            mug = M
        else:
            transfer = min(M - mug, G - glass)
            glass += transfer
            mug -= transfer

    print(glass, mug)

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()

    K = int(data[0])
    G = int(data[1])
    M = int(data[2])

    pour_water(K, G, M)