import sys
from itertools import permutations

def longest_increasing_subsequence(arr):
    dp = [1] * len(arr)
    for i in range(1, len(arr)):
        for j in range(i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def longest_decreasing_subsequence(arr):
    dp = [1] * len(arr)
    for i in range(1, len(arr)):
        for j in range(i):
            if arr[i] < arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def count_valid_permutations(A, B, M):
    AB = A * B
    count = 0
    for perm in permutations(range(1, AB)):
        lis = longest_increasing_subsequence(perm)
        lds = longest_decreasing_subsequence(perm)
        if lis == A and lds == B:
            for n in range(1, AB + 1):
                if n not in perm:
                    new_perm = list(perm) + [n + 0.5]
                    new_lis = longest_increasing_subsequence(new_perm)
                    new_lds = longest_decreasing_subsequence(new_perm)
                    if new_lis == lis and new_lds == lds:
                        count += 1
                        break
    return count % M

def main():
    input = sys.stdin.read()
    A, B, M = map(int, input.split())
    result = count_valid_permutations(A, B, M)
    print(result)

if __name__ == "__main__":
    main()