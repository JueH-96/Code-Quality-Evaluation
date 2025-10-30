def main():
    import sys
    input = sys.stdin.read
    data = input().strip().splitlines()
    
    N = int(data[0])
    people = []
    
    for i in range(1, N + 1):
        name, age = data[i].split()
        age = int(age)
        people.append((name, age))
    
    # Sort people by age to find the youngest
    people.sort(key=lambda x: x[1])
    
    # Find the index of the youngest person
    youngest_index = 0  # since people is sorted, the youngest is at index 0
    
    # Print names starting from the youngest in clockwise order
    for i in range(N):
        print(people[(youngest_index + i) % N][0])

if __name__ == "__main__":
    main()