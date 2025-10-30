from itertools import accumulate

def main():
    N = int(input())
    A = list(map(int, input().split()))
    S = input()

    # Precompute mex values for all possible combinations of two elements
    mex = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if i == j:
                mex[i][j] = 3 - i - j
            else:
                mex[i][j] = 2 - i * j

    # Initialize variables for counting and storing the answer
    count = [0] * 3
    ans = 0

    # Iterate through the sequence
    for i in range(N):
        if S[i] == "M":
            count[A[i]] += 1
        elif S[i] == "E":
            count[A[i]] += 1
            for j in range(3):
                if count[j] > 0:
                    ans += count[j] * mex[A[i]][j]
            count[A[i]] -= 1
        else:  # S[i] == "X"
            count[A[i]] += 1
            for j in range(3):
                if count[j] > 0:
                    for k in range(3):
                        if count[k] > 0:
                            ans += count[k] * mex[j][k]
            count[A[i]] -= 1
            count = [0] * 3

    print(ans)

main()