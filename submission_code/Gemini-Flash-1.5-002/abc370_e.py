# YOUR CODE HERE
MOD = 998244353

def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    count = 0
    for i in range(1 << (n - 1)):
        subsequences = []
        current_subsequence = []
        for j in range(n):
            current_subsequence.append(a[j])
            if j == n - 1 or (i >> j) & 1:
                subsequences.append(current_subsequence)
                current_subsequence = []

        valid = True
        for subsequence in subsequences:
            current_sum = 0
            for x in subsequence:
                current_sum += x
                if current_sum == k:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            count += 1

    print(count % MOD)

solve()