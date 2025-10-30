def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    people = []
    for i in range(1, n + 1):
        parts = data[i].split()
        name = parts[0]
        age = int(parts[1])
        people.append((name, age))
    
    min_index = 0
    min_age = people[0][1]
    for idx in range(1, n):
        if people[idx][1] < min_age:
            min_age = people[idx][1]
            min_index = idx
    
    result = []
    for i in range(n):
        pos = (min_index + i) % n
        result.append(people[pos][0])
    
    for name in result:
        print(name)

if __name__ == "__main__":
    main()