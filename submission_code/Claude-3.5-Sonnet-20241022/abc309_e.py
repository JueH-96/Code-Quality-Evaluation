def get_descendants(person, depth, parents, memo=None):
    if memo is None:
        memo = {}
    
    if depth == 0:
        return {person}
        
    key = (person, depth)
    if key in memo:
        return memo[key]
        
    result = {person}
    for i in range(len(parents)):
        if parents[i] == person:
            result.update(get_descendants(i+2, depth-1, parents, memo))
            
    memo[key] = result
    return result

N, M = map(int, input().split())
parents = list(map(int, input().split()))

covered = set()
for _ in range(M):
    x, y = map(int, input().split())
    covered.update(get_descendants(x, y, parents))
    
print(len(covered))