def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    people = []
    
    for i in range(N):
        parts = input().split()
        name = parts[0]
        age = int(parts[1])
        people.append((name, age))
    
    # Find index of the person with the smallest age
    min_index = 0
    min_age = people[0][1]
    for i in range(1, N):
        if people[i][1] < min_age:
            min_index = i
            min_age = people[i][1]
    
    # Print names in clockwise order starting from the youngest person
    for j in range(N):
        index = (min_index + j) % N
        sys.stdout.write(people[index][0] + "
")

if __name__ == '__main__':
    main()