import math

def solve():
    n = int(input())
    if n <= 0:
        print(0)
        return
    
    result_set = set()
    result_set.add(1)
    
    max_base = int(n**0.5)
    for base in range(2, max_base + 1):
        power = base * base
        while power <= n:
            result_set.add(power)
            if n // power < base: # to avoid potential overflow when calculating next power
                break
            power *= base
            
    print(len(result_set))

if __name__ == '__main__':
    solve()