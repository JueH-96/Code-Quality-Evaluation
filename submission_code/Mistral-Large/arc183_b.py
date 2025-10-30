import sys
input = sys.stdin.read

def can_transform(N, K, A, B):
    for i in range(N):
        if A[i] != B[i]:
            found = False
            for j in range(max(0, i - K), min(N, i + K + 1)):
                if A[j] == B[i]:
                    found = True
                    break
            if not found:
                return "No"
    return "Yes"

def main():
    data = input().split()
    index = 0
    T = int(data[index])
    index += 1
    results = []

    for _ in range(T):
        N = int(data[index])
        K = int(data[index + 1])
        index += 2
        A = list(map(int, data[index:index + N]))
        index += N
        B = list(map(int, data[index:index + N]))
        index += N
        results.append(can_transform(N, K, A, B))

    sys.stdout.write("
".join(results) + "
")

if __name__ == "__main__":
    main()