def main():
    D = input().strip()
    opposite = {
        'N': 'S',
        'S': 'N',
        'E': 'W',
        'W': 'E',
        'NE': 'SW',
        'SW': 'NE',
        'NW': 'SE',
        'SE': 'NW'
    }
    print(opposite[D])

if __name__ == "__main__":
    main()