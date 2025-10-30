def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    results = []
    for sushi_deliciousness in b:
        eater_person = -1
        for person_index in range(n):
            if sushi_deliciousness >= a[person_index]:
                eater_person = person_index + 1
                break
        results.append(eater_person)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()