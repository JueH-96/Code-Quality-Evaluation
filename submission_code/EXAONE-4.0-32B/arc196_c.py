modulo = 998244353
import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
    n = int(data[0])
    s = data[1].strip()
    total_length = 2 * n
    if len(s) < total_length:
        print(0)
        return
    s = s[:total_length]
    
    if s[0] != 'B' or s[-1] != 'W':
        print(0)
        return
        
    ans = 1
    cnt = 0
    for i in range(1, total_length - 1):
        if s[i] == 'W':
            cnt += 1
        else:
            if cnt <= 0:
                print(0)
                return
            cnt -= 1
            ans = (ans * (cnt + 1)) % modulo
            
    print(ans % modulo)

if __name__ == '__main__':
    main()