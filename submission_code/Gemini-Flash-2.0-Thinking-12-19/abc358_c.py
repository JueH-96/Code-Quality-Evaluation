import itertools

def solve():
    n, m = map(int, input().split())
    s = [input() for _ in range(n)]
    flavor_sets = []
    for i in range(n):
        flavors = set()
        for j in range(m):
            if s[i][j] == 'o':
                flavors.add(j + 1)
        flavor_sets.append(flavors)
    
    all_flavors = set(range(1, m + 1))
    
    for k in range(1, n + 1):
        stand_indices = list(range(n))
        for combination_indices in itertools.combinations(stand_indices, k):
            current_flavors = set()
            for index in combination_indices:
                current_flavors.update(flavor_sets[index])
            if current_flavors == all_flavors:
                print(k)
                return
                
if __name__ == '__main__':
    solve()