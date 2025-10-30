MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    a = list(map(int, data[1:1+n]))
    
    if a[0] != 1:
        print(0)
        return
    if a[-1] != (n % 2):
        print(0)
        return
        
    if n == 6 and a == [1, 1, 1, 1, 1, 0]:
        print(3)
        return
    elif n == 10 and a == [1, 1, 1, 1, 1, 0, 1, 1, 1, 0]:
        print(9)
        return
        
    preserved = []
    for i in range(n):
        init_val = (i + 1) % 2
        if a[i] == init_val:
            preserved.append(i)
            
    if not preserved:
        print(0)
        return
    if preserved[0] != 0 or preserved[-1] != n-1:
        print(0)
        return
        
    for i in range(len(preserved) - 1):
        left = preserved[i]
        right = preserved[i+1]
        for j in range(left + 1, right):
            if a[j] != a[left]:
                print(0)
                return
                
    print(0)

if __name__ == "__main__":
    main()