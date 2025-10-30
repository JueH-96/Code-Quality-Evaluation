import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    # Compute the number of mountain folds for each index
    mountain_folds = [0] * (2 ** 100 - A[-1] - 1)
    for i in range(1, 2 ** 100 - A[-1]):
        for j in range(N):
            if (i + A[j]) <= 2 ** 100 - 1:
                if (i + A[j]) % 2 == 0:
                    mountain_folds[i] += 1
    
    # Find the maximum number of mountain folds
    return max(mountain_folds)

print(solve())