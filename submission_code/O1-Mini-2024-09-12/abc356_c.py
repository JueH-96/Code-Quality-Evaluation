# YOUR CODE HERE
import sys
from itertools import combinations

def main():
    import sys
    from sys import stdin
    N, M, K = map(int, stdin.readline().split())
    tests = []
    for _ in range(M):
        parts = stdin.readline().strip().split()
        C_i = int(parts[0])
        keys = list(map(int, parts[1:C_i+1]))
        R_i = parts[-1]
        mask = 0
        for key in keys:
            mask |= 1 << (key-1)
        tests.append( (mask, R_i) )
    
    count = 0
    for comb in range(1<<N):
        valid = True
        real_count = bin(comb).count('1')
        for mask, res in tests:
            intersection = comb & mask
            cnt = bin(intersection).count('1')
            if res == 'o':
                if cnt < K:
                    valid = False
                    break
            else:
                if cnt >= K:
                    valid = False
                    break
        if valid:
            count +=1
    print(count)

if __name__ == "__main__":
    main()