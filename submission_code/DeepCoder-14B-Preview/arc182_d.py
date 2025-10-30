import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    results = []
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        P = list(map(int, input[ptr:ptr+N]))
        ptr += N
        S = sorted(P)
        if P == S:
            results.append(0)
            continue
        
        candidates = []
        for i in range(N):
            if P[i] == S[i]:
                candidates.append(i + 1)  # k is 1-based
        
        found = False
        for k in candidates:
            # Check prefix: P[0..k-2] (0-based) must sort to S[0..k-2]
            prefix = P[:k-1]
            if sorted(prefix) != S[:k-1]:
                continue
            
            # Check suffix: P[k..N-1] must sort to S[k..N-1]
            suffix = P[k:]
            if sorted(suffix) != S[k:]:
                continue
            
            found = True
            break
        
        if found:
            results.append(1)
        else:
            results.append(2)
    
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == '__main__':
    main()