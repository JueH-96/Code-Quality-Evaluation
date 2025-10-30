import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print("No")
        return
        
    n = int(data[0])
    s = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    prefix = [0] * (n+1)
    for i in range(1, n+1):
        prefix[i] = prefix[i-1] + A[i-1]
    total = prefix[n]
    
    seen = set()
    seen.add(0)
    for j in range(1, n+1):
        target = prefix[j] - s
        if target in seen:
            print("Yes")
            return
        seen.add(prefix[j])
        
    prefix_set = set(prefix)
    
    for i in range(0, n+1):
        D = s - (total - prefix[i])
        if D < 0:
            continue
        V = D % total
        if V in prefix_set:
            print("Yes")
            return
            
    print("No")

if __name__ == "__main__":
    main()