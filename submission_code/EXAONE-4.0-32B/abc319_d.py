def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    L = list(map(int, data[2:2+n]))
    
    low = max(L)
    high = sum(L) + (n - 1)
    
    def feasible(W):
        lines = 1
        current = 0
        for num in L:
            if current == 0:
                current = num
            else:
                if current + 1 + num <= W:
                    current += 1 + num
                else:
                    lines += 1
                    current = num
                    if lines > m:
                        return False
        return True
    
    while low < high:
        mid = (low + high) // 2
        if feasible(mid):
            high = mid
        else:
            low = mid + 1
            
    print(low)

if __name__ == '__main__':
    main()