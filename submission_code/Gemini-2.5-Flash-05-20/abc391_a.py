import sys

def main():
    D = sys.stdin.readline().strip()

    opposite_directions = {
        'N': 'S',
        'S': 'N',
        'E': 'W',
        'W': 'E',
        'NE': 'SW',
        'SW': 'NE',
        'NW': 'SE',
        'SE': 'NW'
    }

    print(opposite_directions[D])

if __name__ == '__main__':
    main()