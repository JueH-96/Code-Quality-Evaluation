def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    eaten_by = [-1] * m
    
    for sushi_idx, sushi_deliciousness in enumerate(b):
        for person_idx, person_gourmet_level in enumerate(a):
            if sushi_deliciousness >= person_gourmet_level:
                eaten_by[sushi_idx] = person_idx + 1
                a[person_idx] = float('inf')
                break
    
    for result in eaten_by:
        print(result)

solve()