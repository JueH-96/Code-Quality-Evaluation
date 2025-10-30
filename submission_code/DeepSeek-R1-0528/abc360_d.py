import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    T = int(data[1])
    s = data[2].strip()
    xs = list(map(int, data[3:3+n]))
    
    A = []
    B = []
    
    for i in range(n):
        if s[i] == '1':
            A.append(xs[i])
        else:
            B.append(xs[i])
            
    A.sort()
    B.sort()
    
    j = 0
    k = 0
    ans = 0
    lenB = len(B)
    
    for a in A:
        while j < lenB and B[j] < a:
            j += 1
            
        while k < lenB and B[k] <= a + 2 * T:
            k += 1
            
        ans += (k - j)
        
    print(ans)

if __name__ == "__main__":
    main()