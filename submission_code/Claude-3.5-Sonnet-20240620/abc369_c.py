# YOUR CODE HERE
def count_arithmetic_progressions(N, A):
    count = 0
    for l in range(N):
        r = l
        while r < N:
            if r == l or r == l + 1:
                count += 1
                r += 1
            else:
                if A[r] - A[r-1] == A[r-1] - A[r-2]:
                    count += 1
                    r += 1
                else:
                    break
    return count

N = int(input())
A = list(map(int, input().split()))

result = count_arithmetic_progressions(N, A)
print(result)