import sys

def solve():
    n, m, x1 = map(int, sys.stdin.readline().split())
    trains = []
    for _ in range(m):
        a, b, s, t = map(int, sys.stdin.readline().split())
        trains.append({'departure_city': a, 'arrival_city': b, 'departure_time': s, 'arrival_time': t})
    
    pairs_p = []
    for i in range(m):
        for j in range(m):
            if trains[i]['arrival_city'] == trains[j]['departure_city'] and trains[i]['arrival_time'] <= trains[j]['departure_time']:
                pairs_p.append((i+1, j+1))
                
    dp_weights = [float('-inf')] * m
    dp_weights[0] = 0
    
    for _ in range(m):
        updated = False
        for i, j in pairs_p:
            weight = trains[i-1]['arrival_time'] - trains[j-1]['departure_time']
            if dp_weights[i-1] != float('-inf') and dp_weights[i-1] + weight > dp_weights[j-1]:
                dp_weights[j-1] = dp_weights[i-1] + weight
                updated = True
        if not updated:
            break
            
    x_values = [0] * m
    x_values[0] = x1
    for i in range(1, m):
        if dp_weights[i] == float('-inf'):
            x_values[i] = 0
        else:
            x_values[i] = max(0, x1 + dp_weights[i])
            
    result_x = x_values[1:]
    print(*(int(val) for val in result_x))

if __name__ == '__main__':
    solve()