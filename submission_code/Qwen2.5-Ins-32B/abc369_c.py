def count_arithmetic_subsequences(A):
    N = len(A)
    count = 0
    for i in range(N):
        diff = None
        for j in range(i, N):
            if i == j:
                count += 1
                continue
            if diff is None:
                diff = A[j] - A[i]
            elif A[j] - A[j-1] != diff:
                break
            else:
                count += 1
    return count

if __name__ == "__main__":
    N = int(input())
    A = list(map(int, input().split()))
    print(count_arithmetic_subsequences(A))