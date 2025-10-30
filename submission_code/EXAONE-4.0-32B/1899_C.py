import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    res = []
    for _ in range(t):
        n = int(data[index])
        index += 1
        arr = list(map(int, data[index:index + n]))
        index += n
        
        if n == 0:
            res.append("0")
            continue
            
        current = arr[0]
        best = arr[0]
        
        for i in range(1, n):
            if (arr[i] % 2) != (arr[i - 1] % 2):
                current = max(arr[i], current + arr[i])
            else:
                current = arr[i]
                
            if current > best:
                best = current
                
        res.append(str(best))
    
    print("
".join(res))

if __name__ == "__main__":
    main()