def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        H = int(data[index])
        W = int(data[index + 1])
        index += 2
        
        grid = data[index:index + H]
        index += H
        
        count_A = 0
        count_B = 0
        
        for row in grid:
            count_A += row.count('A')
            count_B += row.count('B')
        
        # Check the parity condition
        if (count_A + 2 * count_B) % 2 == 0:
            # Calculate the number of valid configurations
            result = pow(2, H + W - 1, MOD)
        else:
            result = 0
        
        results.append(result)
    
    for res in results:
        print(res)