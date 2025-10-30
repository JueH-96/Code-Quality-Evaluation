import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    n = int(input[ptr])
    ptr += 1
    m = int(input[ptr])
    ptr += 1
    x1 = int(input[ptr])
    ptr += 1
    
    trains = []
    for k in range(m):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        s = int(input[ptr])
        ptr += 1
        t = int(input[ptr])
        ptr += 1
        trains.append((t, k + 1, a, b, s))
    
    sorted_trains = sorted(trains, key=lambda x: x[0])
    
    X = [0] * (m + 1)
    X[1] = x1
    
    city_data = dict()
    
    for train in sorted_trains:
        t_arrival, idx, a, b, s_departure = train
        
        if idx == 1:
            arrival_city = b
            x_plus_t = X[1] + t_arrival
            if arrival_city not in city_data:
                city_data[arrival_city] = ([], [])
            times, maxes = city_data[arrival_city]
            times.append(t_arrival)
            if maxes:
                new_max = max(maxes[-1], x_plus_t)
            else:
                new_max = x_plus_t
            maxes.append(new_max)
        else:
            departure_city = a
            max_val = -float('inf')
            if departure_city in city_data:
                times, maxes = city_data[departure_city]
                s = s_departure
                idx_city = bisect.bisect_right(times, s)
                if idx_city > 0:
                    max_val = maxes[idx_city - 1]
            candidate = max_val - s_departure
            X_val = max(0, candidate)
            X[idx] = X_val
            
            arrival_city = b
            x_plus_t = X_val + t_arrival
            if arrival_city not in city_data:
                city_data[arrival_city] = ([], [])
            times_city, maxes_city = city_data[arrival_city]
            times_city.append(t_arrival)
            if maxes_city:
                new_max = max(maxes_city[-1], x_plus_t)
            else:
                new_max = x_plus_t
            maxes_city.append(new_max)
    
    print(' '.join(map(str, X[2:m+1])))

if __name__ == '__main__':
    main()