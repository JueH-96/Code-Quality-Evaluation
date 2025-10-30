def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1
    from collections import Counter
    
    outputs = []
    for _ in range(t):
        n = int(input_data[idx])
        idx += 1
        a = list(map(int, input_data[idx:idx+n]))
        idx += n
        
        freq = Counter(a)
        
        # 1) Count all pairs (i < j) where a_i = a_j, because that implies 2^(a_i) ^ 2^(a_j) == 2^(a_j) ^ 2^(a_i)
        #    Each value v in freq contributes freq[v]*(freq[v]-1)//2 such pairs.
        result = 0
        for v in freq:
            c = freq[v]
            result += c * (c - 1) // 2
        
        # 2) Count pairs where {a_i, a_j} = {1, 2}, because c(1) = 1/2 and c(2) = 1/2,
        #    so 2^(1)^2^(2) = 2^(2)^2^(1).
        result += freq[1] * freq[2]
        
        outputs.append(str(result))
    
    print("
".join(outputs))

def main():
    solve()

if __name__ == "__main__":
    main()