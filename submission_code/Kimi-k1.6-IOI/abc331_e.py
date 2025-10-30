import math

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    m = int(data[idx+1])
    l = int(data[idx+2])
    idx +=3
    
    a = list(map(int, data[idx:idx+n]))
    idx +=n
    b = list(map(int, data[idx:idx+m]))
    idx +=m
    
    forbidden = set()
    for _ in range(l):
        c = int(data[idx])
        d = int(data[idx+1])
        forbidden.add((c, d))
        idx +=2
    
    # Sort a and b with their original indices (1-based)
    a_sorted = sorted([(a[i], i+1) for i in range(n)], key=lambda x: (-x[0], x[1]))
    b_sorted = sorted([(b[i], i+1) for i in range(m)], key=lambda x: (-x[0], x[1]))
    
    k = max(1, int(math.sqrt(l)) + 1)
    k_a = min(k, n)
    k_b = min(k, m)
    
    a_candidates = a_sorted[:k_a]
    b_candidates = b_sorted[:k_b]
    
    max_sum = -float('inf')
    
    for aval, aidx in a_candidates:
        for bval, bidx in b_candidates:
            if (aidx, bidx) not in forbidden:
                current_sum = aval + bval
                if current_sum > max_sum:
                    max_sum = current_sum
    
    print(max_sum)

if __name__ == "__main__":
    main()