def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353

    ans = [0] * n
    
    for i in range(1, n + 1):
        count = 0
        
        def is_arithmetic(seq):
            if len(seq) < 2:
                return True
            diff = seq[1] - seq[0]
            for j in range(2, len(seq)):
                if seq[j] - seq[j-1] != diff:
                    return False
            return True

        def find_subsequences(index, current_subsequence):
            nonlocal count
            if len(current_subsequence) == i:
                if is_arithmetic(current_subsequence):
                    count = (count + 1) % mod
                return

            if index == n:
                return

            find_subsequences(index + 1, current_subsequence)
            find_subsequences(index + 1, current_subsequence + [a[index]])

        find_subsequences(0, [])
        ans[i-1] = count

    print(*ans)

solve()