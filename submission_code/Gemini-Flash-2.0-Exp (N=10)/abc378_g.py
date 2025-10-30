def solve():
    a, b, m = map(int, input().split())
    n = a * b - 1
    
    dp = {}

    def count_permutations(arr, lis_len, lds_len):
        if not arr:
            return 1 if lis_len == a and lds_len == b else 0
        
        state = (tuple(arr), lis_len, lds_len)
        if state in dp:
            return dp[state]
        
        count = 0
        for i in range(len(arr)):
            
            
            new_arr = arr[:i] + arr[i+1:]
            
            
            new_lis_len = lis_len
            if not any(arr[i] > arr[j] for j in range(i)):
                new_lis_len = max(lis_len, 1)
                
                
                temp_lis = []
                for x in arr[:i+1]:
                    
                    if not temp_lis or x > temp_lis[-1]:
                        temp_lis.append(x)
                    else:
                        
                        l, r = 0, len(temp_lis) - 1
                        while l <= r:
                            mid = (l + r) // 2
                            if temp_lis[mid] < x:
                                l = mid + 1
                            else:
                                r = mid - 1
                        temp_lis[l] = x
                new_lis_len = max(new_lis_len, len(temp_lis))
            
            new_lds_len = lds_len
            if not any(arr[i] < arr[j] for j in range(i)):
                new_lds_len = max(lds_len, 1)
                
                temp_lds = []
                for x in arr[:i+1]:
                    if not temp_lds or x < temp_lds[-1]:
                        temp_lds.append(x)
                    else:
                        l, r = 0, len(temp_lds) - 1
                        while l <= r:
                            mid = (l + r) // 2
                            if temp_lds[mid] > x:
                                l = mid + 1
                            else:
                                r = mid - 1
                        temp_lds[l] = x
                new_lds_len = max(new_lds_len, len(temp_lds))
            
            
            
            
            count = (count + count_permutations(new_arr, new_lis_len, new_lds_len)) % m
        
        dp[state] = count
        return count

    nums = list(range(1, n + 1))
    result = count_permutations(nums, 0, 0)
    print(result)

solve()