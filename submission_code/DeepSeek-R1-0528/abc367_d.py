import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    n = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+n]))
    total = sum(A)
    P = [0] * n
    for i in range(1, n):
        P[i] = P[i-1] + A[i-1]
    
    residues = [p % M for p in P]
    T = total % M
    
    freq = [0] * M
    for r in residues:
        freq[r] += 1
        
    case1 = 0
    for count in freq:
        case1 += count * (count - 1) // 2
        
    freq2 = [0] * M
    for r in residues:
        freq2[r] += 1
        
    case2 = 0
    for j in range(n):
        r_val = residues[j]
        freq2[r_val] -= 1
        target = (r_val + T) % M
        case2 += freq2[target]
        
    ans = case1 + case2
    print(ans)

if __name__ == "__main__":
    main()