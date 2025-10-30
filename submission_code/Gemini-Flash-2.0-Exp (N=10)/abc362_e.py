def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353
    
    for k in range(1, n + 1):
        count = 0
        
        def find_subsequences(index, current_subsequence):
            nonlocal count
            if len(current_subsequence) == k:
                if len(current_subsequence) <= 1:
                    count = (count + 1) % mod
                    return
                
                diff = current_subsequence[1] - current_subsequence[0]
                is_arithmetic = True
                for i in range(2, len(current_subsequence)):
                    if current_subsequence[i] - current_subsequence[i-1] != diff:
                        is_arithmetic = False
                        break
                if is_arithmetic:
                    count = (count + 1) % mod
                return
            
            if index == n:
                return
            
            find_subsequences(index + 1, current_subsequence)
            find_subsequences(index + 1, current_subsequence + [a[index]])
            
        find_subsequences(0, [])
        print(count, end=" ")
    print()

solve()