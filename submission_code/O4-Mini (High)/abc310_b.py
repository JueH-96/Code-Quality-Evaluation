def main():
    import sys
    input = sys.stdin.readline
    N, M = map(int, input().split())
    P = []
    F = []
    for _ in range(N):
        data = list(map(int, input().split()))
        P.append(data[0])
        F.append(set(data[2:]))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Check if product j is strictly superior to product i
            if P[i] >= P[j] and F[i].issubset(F[j]) and (P[i] > P[j] or len(F[i]) < len(F[j])):
                print("Yes")
                return
    print("No")

main()