import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    s_list = [sys.stdin.readline().strip() for _ in range(n)]
    stands_flavors = []
    for i in range(n):
        flavors = []
        for j in range(m):
            if s_list[i][j] == 'o':
                flavors.append(j+1)
        stands_flavors.append(set(flavors))
    
    all_flavors = set(range(1, m + 1))
    
    for k in range(1, n + 1):
        import itertools
        stand_indices = list(range(n))
        combinations = itertools.combinations(stand_indices, k)
        for combination_indices in combinations:
            current_flavors = set()
            for index in combination_indices:
                current_flavors.update(stands_flavors[index])
            if current_flavors == all_flavors:
                print(k)
                return
                
    print(n) # Should not reach here based on problem constraints

if __name__ == '__main__':
    solve()