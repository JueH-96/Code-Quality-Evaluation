import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    L = list(map(int, data[2:2+n]))
    
    if n == 0:
        print(0)
        return
        
    low = max(L)
    total = sum(L)
    high = total + (n - 1)
    
    while low < high:
        mid = (low + high) // 2
        lines = 1
        current = L[0]
        valid = True
        for i in range(1, n):
            if current + 1 + L[i] <= mid:
                current += 1 + L[i]
            else:
                lines += 1
                if lines > m:
                    valid = False
                    break
                current = L[i]
        
        if valid:
            high = mid
        else:
            low = mid + 1
            
    print(low)

if __name__ == "__main__":
    main()