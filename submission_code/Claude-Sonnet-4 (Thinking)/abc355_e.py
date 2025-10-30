import sys

def solve():
    N, L, R = map(int, input().split())
    
    total = 0
    pos = L
    
    while pos <= R:
        remaining = R - pos + 1
        
        # Find the largest power of 2 that divides pos
        if pos == 0:
            max_alignment = N
        else:
            max_alignment = 0
            temp = pos
            while temp % 2 == 0:
                temp //= 2
                max_alignment += 1
        
        # Find the largest power of 2 <= remaining
        max_length_power = 0
        while (1 << (max_length_power + 1)) <= remaining:
            max_length_power += 1
        
        k = min(max_alignment, max_length_power)
        
        # Query for segment [pos, pos + 2^k - 1]
        j = pos // (1 << k)
        print(f"? {k} {j}")
        sys.stdout.flush()
        result = int(input())
        if result == -1:
            return
        
        total = (total + result) % 100
        pos += (1 << k)
    
    print(f"! {total}")
    sys.stdout.flush()

solve()