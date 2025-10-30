def main():
    import sys
    N = int(sys.stdin.read().strip())
    
    # Water stations exist every 5 km, including 0 and 100.
    # Compute the nearest water station by simply checking multiples of 5 from 0 to 100.
    nearest_station = None
    min_distance = float('inf')
    for station in range(0, 101, 5):
        distance = abs(N - station)
        if distance < min_distance:
            min_distance = distance
            nearest_station = station
            
    print(nearest_station)
    
if __name__ == '__main__':
    main()