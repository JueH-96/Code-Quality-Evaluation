def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    for sushi_deliciousness in b:
        eaten = False
        for person_id in range(n):
            if sushi_deliciousness >= a[person_id]:
                print(person_id + 1)
                a[person_id] = float('inf')
                eaten = True
                break
        if not eaten:
            print(-1)

solve()