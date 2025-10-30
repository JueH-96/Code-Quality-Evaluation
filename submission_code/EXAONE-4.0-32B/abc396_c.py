import sys

def main():
    data = sys.stdin.read().strip().split()
    n = int(data[0])
    m = int(data[1])
    B = list(map(int, data[2:2+n]))
    W = list(map(int, data[2+n:2+n+m]))
    
    B.sort(reverse=True)
    W.sort(reverse=True)
    
    prefixB = [0] * (n+1)
    for i in range(1, n+1):
        prefixB[i] = prefixB[i-1] + B[i-1]
        
    prefixW = [0] * (m+1)
    for i in range(1, m+1):
        prefixW[i] = prefixW[i-1] + W[i-1]
        
    count_non_neg = 0
    for num in B:
        if num >= 0:
            count_non_neg += 1
        else:
            break
            
    max_w = min(n, m)
    best = -10**20
    
    for w in range(0, max_w+1):
        if w <= count_non_neg:
            s_black = prefixB[count_non_neg]
        else:
            s_black = prefixB[w]
        s_white = prefixW[w]
        total = s_black + s_white
        if total > best:
            best = total
            
    print(best)

if __name__ == "__main__":
    main()