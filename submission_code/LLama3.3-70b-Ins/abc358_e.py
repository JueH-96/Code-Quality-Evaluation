import sys
from itertools import product

MOD = 998244353

def count_strings(K, C):
    count = 0
    for length in range(1, K + 1):
        for chars in product(range(26), repeat=length):
            valid = True
            for i in range(26):
                if chars.count(i) > C[i]:
                    valid = False
                    break
            if valid:
                count += 1
    return count % MOD

def main():
    K = int(sys.stdin.readline().strip())
    C = list(map(int, sys.stdin.readline().strip().split()))
    result = count_strings(K, C)
    print(result)

if __name__ == "__main__":
    main()