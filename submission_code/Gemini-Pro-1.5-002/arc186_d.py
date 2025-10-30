def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    mod = 998244353

    dp = {}

    def is_polish(arr):
        if tuple(arr) in dp:
            return dp[tuple(arr)]
        
        m = len(arr)
        if m == 0:
            return False
        
        if m == 1 and arr[0] == 0:
            dp[tuple(arr)] = True
            return True

        if arr[0] >= m:
            dp[tuple(arr)] = False
            return False
        
        if arr[0] == 0 and m > 1:
            dp[tuple(arr)] = False
            return False

        
        for i in range(1 << (m - 1)):
            partitions = []
            current_partition = []
            for j in range(m - 1):
                current_partition.append(arr[j+1])
                if (i >> j) & 1 or j == m - 2:
                    partitions.append(current_partition)
                    current_partition = []
            
            if len(partitions) != arr[0]:
                continue

            polish = True
            for partition in partitions:
                if not is_polish(partition):
                    polish = False
                    break
            
            if polish:
                dp[tuple(arr)] = True
                return True
        
        dp[tuple(arr)] = False
        return False

    def count_polish(length, prefix):
        if length == 0:
            return 1 if is_polish(prefix) else 0
        
        count = 0
        for i in range(n if len(prefix) == 0 else (a[len(prefix)] + 1)):
            new_prefix = prefix + [i]
            if len(new_prefix) > 1 and new_prefix[0] >= len(new_prefix):
                break
            if len(new_prefix) > 1 and new_prefix[0] == 0:
                break
            
            if len(prefix) < n and i < a[len(prefix)]:
                if is_polish(new_prefix):
                    count = (count + pow(n, length - 1, mod)) % mod
            elif len(prefix) < n and i == a[len(prefix)]:
                count = (count + count_polish(length - 1, new_prefix)) % mod
            elif len(prefix) == n:
                if is_polish(new_prefix):
                    count = (count + 1) % mod
        return count

    print(count_polish(n, []))

solve()