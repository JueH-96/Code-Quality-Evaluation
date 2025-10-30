import sys

def main():
    N = int(sys.stdin.readline())
    # Round N to the nearest multiple of 5.
    # Adding 2 shifts the cut-off so that remainders 0,1,2 go down,
    # and 3,4 go up, guaranteeing the nearest station.
    nearest_station = ((N + 2) // 5) * 5
    print(nearest_station)

if __name__ == "__main__":
    main()