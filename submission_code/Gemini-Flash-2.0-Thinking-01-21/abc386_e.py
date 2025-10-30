import itertools

def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    max_xor_sum = 0
    combinations = itertools.combinations(a, k)
    
    for combination in combinations:
        current_xor = 0
        for element in combination:
            current_xor ^= element
        max_xor_sum = max(max_xor_sum, current_xor)
        
    print(max_xor_sum)

if __name__ == '__main__':
    solve()