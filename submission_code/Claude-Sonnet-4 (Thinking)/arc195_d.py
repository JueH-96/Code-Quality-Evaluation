def solve(arr):
    memo = {}
    
    def dp(arr_tuple):
        if not arr_tuple:
            return 0
        if arr_tuple in memo:
            return memo[arr_tuple]
        
        arr = list(arr_tuple)
        n = len(arr)
        result = float('inf')
        
        # Try deleting each possible prefix of length i
        for i in range(1, n + 1):
            # Try making arr[0:i] all equal to each value that exists in the array
            values = set(arr)
            for value in values:
                # Check if there are enough occurrences of this value
                if arr.count(value) < i:
                    continue
                
                # Calculate cost to make arr[0:i] all equal to value
                temp_arr = arr[:]
                cost = 0
                
                # Bring i occurrences of 'value' to positions 0,1,...,i-1
                for j in range(i):
                    if temp_arr[j] != value:
                        # Find the nearest occurrence of 'value' to the right
                        for k in range(j + 1, n):
                            if temp_arr[k] == value:
                                # Swap it to position j using adjacent swaps
                                while k > j:
                                    temp_arr[k], temp_arr[k - 1] = temp_arr[k - 1], temp_arr[k]
                                    k -= 1
                                    cost += 1
                                break
                
                # Delete the prefix and solve recursively
                remaining = tuple(temp_arr[i:])
                total_cost = cost + 1 + dp(remaining)
                result = min(result, total_cost)
        
        memo[arr_tuple] = result
        return result
    
    return dp(tuple(arr))

# Read input and solve
T = int(input())
for _ in range(T):
    N = int(input())
    A = list(map(int, input().split()))
    print(solve(A))