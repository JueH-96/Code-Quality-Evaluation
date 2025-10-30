import sys

def main():
    MAX_VAL = 1500000
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    size_bit = MAX_VAL
    bit = [0] * (size_bit + 2)
    
    def update(idx, delta):
        while idx <= size_bit:
            bit[idx] += delta
            idx += idx & -idx
            
    def prefix_query(idx):
        if idx == 0:
            return 0
        s = 0
        while idx:
            s += bit[idx]
            idx -= idx & -idx
        return s

    F = [0] * (n + 1)
    
    for i in range(1, n + 1):
        if i - 1 > 0:
            prev_count = prefix_query(i - 1)
        else:
            prev_count = 0
        F[i] = (i - 1) - prev_count
        x = A[i - 1] + F[i] + i
        if x <= size_bit:
            update(x, 1)
        else:
            update(size_bit, 1)
            
    res = []
    for i in range(1, n + 1):
        total_here = A[i - 1] + F[i]
        give_count = n - i
        final_stones = total_here - give_count
        if final_stones < 0:
            final_stones = 0
        res.append(str(final_stones))
        
    print(" ".join(res))

if __name__ == "__main__":
    main()