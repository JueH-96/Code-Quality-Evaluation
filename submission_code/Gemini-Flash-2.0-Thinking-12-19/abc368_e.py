import sys

def solve():
    n, m, x1 = map(int, sys.stdin.readline().split())
    trains = []
    for _ in range(m):
        a, b, s, t = map(int, sys.stdin.readline().split())
        trains.append({'departure_city': a, 'arrival_city': b, 'departure_time': s, 'arrival_time': t})
    
    constraints = []
    for i in range(m):
        for j in range(m):
            if trains[i]['arrival_city'] == trains[j]['departure_city'] and trains[i]['arrival_time'] <= trains[j]['departure_time']:
                constraints.append({'train1_index': i, 'train2_index': j, 'cost': trains[j]['departure_time'] - trains[i]['arrival_time']})
                
    x = [0] * m
    x[0] = x1
    
    for _ in range(m):
        updated = False
        for constraint in constraints:
            i_index = constraint['train1_index']
            j_index = constraint['train2_index']
            cost = constraint['cost']
            if x[j_index] < x[i_index] - cost:
                x[j_index] = x[i_index] - cost
                updated = True
        if not updated:
            break
            
    for i in range(1, m):
        x[i] = max(0, x[i])
        
    result_x = x[1:]
    print(*(result_x))

if __name__ == '__main__':
    solve()