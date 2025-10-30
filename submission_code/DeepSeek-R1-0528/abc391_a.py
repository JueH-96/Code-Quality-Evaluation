def main():
    opposite_map = {
        'N': 'S',
        'S': 'N',
        'E': 'W',
        'W': 'E',
        'NE': 'SW',
        'NW': 'SE',
        'SE': 'NW',
        'SW': 'NE'
    }
    
    D = input().strip()
    print(opposite_map[D])

if __name__ == "__main__":
    main()