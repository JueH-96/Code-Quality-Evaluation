def check_all_flavors(selected_stands, stands, M):
    # Check if selected stands cover all flavors
    flavors = [False] * M
    for stand_idx in selected_stands:
        for j in range(M):
            if stands[stand_idx][j] == 'o':
                flavors[j] = True
    return all(flavors)

def solve():
    N, M = map(int, input().split())
    stands = []
    for _ in range(N):
        stands.append(input())
    
    # Try all possible combinations of stands from 1 to N stands
    for num_stands in range(1, N+1):
        # Try all possible combinations of num_stands stands
        from itertools import combinations
        for comb in combinations(range(N), num_stands):
            if check_all_flavors(comb, stands, M):
                print(num_stands)
                return

solve()