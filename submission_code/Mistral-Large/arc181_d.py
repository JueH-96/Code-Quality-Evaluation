import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0

    N = int(data[index])
    index += 1
    P = list(map(int, data[index:index+N]))
    index += N
    M = int(data[index])
    index += 1
    A = list(map(int, data[index:index+M]))

    # Initialize inversion count
    inversion_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if P[i] > P[j]:
                inversion_count += 1

    # Process each operation
    results = []
    for k in range(M):
        op = A[k]
        for i in range(op - 1):
            if P[i] > P[i + 1]:
                P[i], P[i + 1] = P[i + 1], P[i]
                # Update inversion count
                if P[i] > P[i + 1]:
                    inversion_count -= 1
                else:
                    inversion_count += 1
        results.append(inversion_count)

    # Output results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()