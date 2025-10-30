def count_inversions(P):
    inversions = 0
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if P[i] > P[j]:
                inversions += 1
    return inversions

def perform_operations(P, A):
    for k in A:
        for i in range(k - 1):
            if P[i] > P[i + 1]:
                P[i], P[i + 1] = P[i + 1], P[i]
        yield count_inversions(P)

def main():
    N = int(input().strip())
    P = list(map(int, input().strip().split()))
    M = int(input().strip())
    A = list(map(int, input().strip().split()))

    results = perform_operations(P, A)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()