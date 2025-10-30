MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    
    if n == 0:
        print(0)
        return
    
    # Check first and last elements
    if a[0] != 1:
        print(0)
        return
    if a[-1] != (n % 2):
        print(0)
        return
    
    # Check runs and count K
    if n == 1:
        print(1)
        return
    
    k = 1
    prev = a[0]
    for i in range(1, n):
        if a[i] == prev:
            print(0)
            return
        k += 1
        prev = a[i]
    
    # Calculate m
    if (n - k) % 2 != 0 or (n - k) < 0:
        print(0)
        return
    m = (n - k) // 2
    
    if m == 0:
        print(1)
    else:
        print(pow(3, m-1, MOD))

if __name__ == '__main__':
    main()