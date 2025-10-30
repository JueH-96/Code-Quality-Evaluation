# YOUR CODE HERE
def nearest_water_station(N):
    # Calculate the nearest water station by rounding to the nearest multiple of 5
    nearest_station = round(N / 5) * 5
    return nearest_station

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    print(nearest_water_station(N))