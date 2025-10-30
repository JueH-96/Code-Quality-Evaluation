def main():
    points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    dists = [3, 1, 4, 1, 5, 9]
    
    cumulative = {}
    cumulative['A'] = 0
    current = 0
    for i in range(len(dists)):
        current += dists[i]
        cumulative[points[i+1]] = current
        
    p, q = input().split()
    distance = abs(cumulative[p] - cumulative[q])
    print(distance)

if __name__ == "__main__":
    main()