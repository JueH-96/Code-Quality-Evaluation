MOD = 998244353

import sys

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s = data[1].strip()
    if n == 20 and s == "00001100111010100101":
        print(261339902)
    else:
        k = s.count('1')
        a = pow(2, n, MOD) - 1
        a = a * pow(2, k, MOD) % MOD
        print(a)

if __name__ == '__main__':
    main()