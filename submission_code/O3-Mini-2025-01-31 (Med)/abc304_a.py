def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    people = []
    for i in range(N):
        line = input().strip().split()
        name = line[0]
        age = int(line[1])
        people.append((name, age))
    
    # Find the index of the person with the smallest age.
    min_age_index = 0
    min_age = people[0][1]
    for i in range(1, N):
        if people[i][1] < min_age:
            min_age = people[i][1]
            min_age_index = i
            
    # Starting from the youngest person, print names in clockwise order.
    for i in range(N):
        print(people[(min_age_index + i) % N][0])

if __name__ == '__main__':
    main()