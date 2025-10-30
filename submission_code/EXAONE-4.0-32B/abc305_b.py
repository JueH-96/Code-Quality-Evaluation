def main():
    data = input().split()
    p = data[0]
    q = data[1]
    
    cumulative = {
        'A': 0,
        'B': 3,
        'C': 4,
        'D': 8,
        'E': 9,
        'F': 14,
        'G': 23
    }
    
    distance = abs(cumulative[p] - cumulative[q])
    print(distance)

if __name__ == '__main__':
    main()