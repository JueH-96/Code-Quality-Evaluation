import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    idx += 1
    q = int(data[idx])
    idx += 1
    
    R = list(map(int, data[idx:idx+n]))
    idx += n
    
    R.sort()
    
    prefix = [0]
    current = 0
    for num in R:
        current += num
        prefix.append(current)
    
    for _ in range(q):
        x = int(data[idx])
        idx += 1
        pos = bisect.bisect_right(prefix, x)
        print(pos - 1)
        
if __name__ == '__main__':
    main()