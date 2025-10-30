import sys
input = sys.stdin.read

def find_triple(N, X, A):
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                if A[i] + A[j] + A[k] == X:
                    return i + 1, j + 1, k + 1
    return -1

def main():
    data = input().split()
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:]))

    result = find_triple(N, X, A)
    if result == -1:
        print(-1)
    else:
        print(result[0], result[1], result[2])

if __name__ == "__main__":
    main()