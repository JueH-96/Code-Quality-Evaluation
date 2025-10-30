def solve():
    n, m = map(int, input().split())
    noodles = [0] * n
    row = list(range(1, n + 1))
    returns = {}

    for _ in range(m):
        t, w, s = map(int, input().split())
        
        
        available_row = []
        for person in row:
            if person not in returns or returns[person] <= t:
                available_row.append(person)
        
        if available_row:
            person_at_front = available_row[0]
            noodles[person_at_front - 1] += w
            row.remove(person_at_front)
            returns[person_at_front] = t + s
            
    for amount in noodles:
        print(amount)

solve()