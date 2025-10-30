import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    n, T = map(int, data[0].split())
    s = data[1].strip()
    xs = list(map(int, data[2].split()))
    
    ants = []
    for i in range(n):
        d = 1 if s[i] == '1' else 0
        ants.append((xs[i], d))
        
    ants.sort(key=lambda x: x[0])
    
    right_ants = [x for x, d in ants if d == 1]
    
    j = 0
    k = 0
    ans = 0
    for x, d in ants:
        while j < len(right_ants) and right_ants[j] <= x:
            j += 1
        while k < j and right_ants[k] < x - 2 * T:
            k += 1
        if d == 0:
            ans += j - k
            
    print(ans)

if __name__ == '__main__':
    main()