import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
        
    n = int(data[0])
    m = int(data[1])
    B = list(map(int, data[2:2+n]))
    W = list(map(int, data[2+n:2+n+m]))
    
    B.sort(reverse=True)
    p = 0
    for x in B:
        if x >= 0:
            p += 1
        else:
            break
            
    prefixB = [0] * (n+1)
    for i in range(1, n+1):
        prefixB[i] = prefixB[i-1] + B[i-1]
        
    W.sort(reverse=True)
    prefixW = [0] * (m+1)
    for i in range(1, m+1):
        prefixW[i] = prefixW[i-1] + W[i-1]
        
    min_val = min(n, m)
    ans = 0
    for k in range(0, min_val+1):
        if k <= p:
            black_sum = prefixB[p]
        else:
            black_sum = prefixB[k]
            
        white_sum = prefixW[k]
        total = black_sum + white_sum
        if total > ans:
            ans = total
            
    print(ans)

if __name__ == '__main__':
    main()