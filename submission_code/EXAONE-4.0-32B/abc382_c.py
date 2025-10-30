import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+n]))
    B = list(map(int, data[2+n:2+n+m]))
    
    max_val = 200000
    arr = [n + 1] * (max_val + 1)
    
    for i in range(n):
        a_val = A[i]
        if i + 1 < arr[a_val]:
            arr[a_val] = i + 1
            
    min_idx = [0] * (max_val + 1)
    min_idx[0] = arr[0]
    for x in range(1, max_val + 1):
        min_idx[x] = min(min_idx[x - 1], arr[x])
        
    results = []
    for b in B:
        if min_idx[b] > n:
            results.append("-1")
        else:
            results.append(str(min_idx[b]))
            
    print("
".join(results))

if __name__ == "__main__":
    main()