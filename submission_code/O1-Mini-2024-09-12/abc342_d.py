import sys
import math
from collections import defaultdict

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    
    Z = A.count(0)
    non_zero = [x for x in A if x !=0]
    
    max_A = max(non_zero) if non_zero else 0
    sieve = [0]*(max_A +1)
    for i in range(2, max_A+1):
        if sieve[i]==0:
            sieve[i]=i
            for j in range(i*i, max_A+1, i):
                if sieve[j]==0:
                    sieve[j]=i
    def get_square_free(x):
        s=1
        while x >1:
            p = sieve[x] if sieve[x]!=0 else x
            cnt=0
            while x % p ==0:
                cnt +=1
                x =x//p
            if cnt %2 ==1:
                s *=p
        return s
    counts = defaultdict(int)
    for num in non_zero:
        if num ==1:
            s=1
        else:
            s = get_square_free(num)
        counts[s] +=1
    total = Z*(N - Z) + Z*(Z-1)//2
    for cnt in counts.values():
        total += cnt*(cnt-1)//2
    print(total)

if __name__ == "__main__":
    main()