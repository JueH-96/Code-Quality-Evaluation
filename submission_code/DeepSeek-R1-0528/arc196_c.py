MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    s = data[1].strip()
    
    if s[0] == 'W' or s[-1] == 'B':
        print(0)
        return
        
    ans = 1
    a = 0
    for char in s:
        if char == 'B':
            a += 1
        else:
            if a == 0:
                print(0)
                return
            ans = (ans * a) % MOD
            a -= 1
            
    if a != 0:
        print(0)
    else:
        print(ans)

if __name__ == '__main__':
    main()