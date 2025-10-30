import itertools
import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    A = list(map(int, data[3:3+n]))
    B = list(map(int, data[3+n:3+2*n]))
    
    for i in range(n):
        if A[i] > X or B[i] > Y:
            print(1)
            return
            
    totalA = sum(A)
    totalB = sum(B)
    if totalA <= X and totalB <= Y:
        print(n)
        return
        
    if n <= 20:
        indices = list(range(n))
        for k in range(2, n+1):
            for j in range(n):
                others = [i for i in range(n) if i != j]
                for comb in itertools.combinations(others, k-1):
                    totalA_comb = 0
                    totalB_comb = 0
                    for idx in comb:
                        totalA_comb += A[idx]
                        totalB_comb += B[idx]
                    if totalA_comb <= X and totalB_comb <= Y:
                        totalA_with_j = totalA_comb + A[j]
                        totalB_with_j = totalB_comb + B[j]
                        if totalA_with_j > X or totalB_with_j > Y:
                            print(k)
                            return
    sorted_A = sorted(A, reverse=True)
    sorted_B = sorted(B, reverse=True)
    k1 = None
    total = 0
    for i in range(n):
        total += sorted_A[i]
        if total > X:
            k1 = i+1
            break
    k2 = None
    total = 0
    for i in range(n):
        total += sorted_B[i]
        if total > Y:
            k2 = i+1
            break
            
    if k1 is not None and k2 is not None:
        ans = min(k1, k2)
    elif k1 is not None:
        ans = k1
    elif k2 is not None:
        ans = k2
    else:
        ans = n
    print(ans)

if __name__ == "__main__":
    main()