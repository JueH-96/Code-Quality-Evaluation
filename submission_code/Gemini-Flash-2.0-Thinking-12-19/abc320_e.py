def solve():
    n, m = map(int, input().split())
    events = []
    for _ in range(m):
        events.append(list(map(int, input().split())))
    
    row = list(range(1, n + 1))
    out_of_row = {}
    noodles_received = [0] * (n + 1)
    
    for event in events:
        t_i, w_i, s_i = event
        
        returned_persons = []
        for person_id, return_time in list(out_of_row.items()):
            if return_time <= t_i:
                returned_persons.append(person_id)
                
        for person_id in returned_persons:
            row.append(person_id)
            del out_of_row[person_id]
            
        row.sort()
        
        if row:
            person_at_front = row.pop(0)
            noodles_received[person_at_front] += w_i
            out_of_row[person_at_front] = t_i + s_i
            
    for i in range(1, n + 1):
        print(noodles_received[i])

if __name__ == '__main__':
    solve()