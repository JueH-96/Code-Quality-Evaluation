import sys

def main() -> None:
    d = sys.stdin.readline().strip()          # read the direction
    opposite = {                              # map to its opposite
        'N':  'S', 'S':  'N',
        'E':  'W', 'W':  'E',
        'NE': 'SW', 'SW': 'NE',
        'NW': 'SE', 'SE': 'NW'
    }
    print(opposite[d])

if __name__ == "__main__":
    main()