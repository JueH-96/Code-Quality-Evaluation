def solve():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    idx = 1
    
    results = []
    for _ in range(t):
        n = int(data[idx]); idx += 1
        arr = list(map(int, data[idx:idx+n]))
        idx += n
        
        # dpEven and dpOdd hold the maximum sum of a valid subarray ending at
        # the current position with an even or odd element respectively.
        dpEven = float('-inf')
        dpOdd = float('-inf')
        
        max_sum = float('-inf')
        
        for x in arr:
            if x % 2 == 0:  # x is even
                newEven = max(x, (dpOdd + x) if dpOdd != float('-inf') else x)
                newOdd = float('-inf')
            else:  # x is odd
                newEven = float('-inf')
                newOdd = max(x, (dpEven + x) if dpEven != float('-inf') else x)
            
            dpEven, dpOdd = newEven, newOdd
            max_sum = max(max_sum, dpEven, dpOdd)
        
        results.append(str(max_sum))
    
    print("
".join(results))

def main():
    solve()

if __name__ == "__main__":
    main()