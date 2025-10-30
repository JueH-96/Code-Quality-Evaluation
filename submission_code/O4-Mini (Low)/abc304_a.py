def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    people = []
    for _ in range(N):
        line = input().split()
        name = line[0]
        age = int(line[1])
        people.append((name, age))
    
    # Find the index of the youngest person
    min_idx = 0
    min_age = people[0][1]
    for i in range(1, N):
        if people[i][1] < min_age:
            min_age = people[i][1]
            min_idx = i
    
    # Print in clockwise order starting from the youngest
    for i in range(N):
        idx = (min_idx + i) % N
        print(people[idx][0])

if __name__ == "__main__":
    main()