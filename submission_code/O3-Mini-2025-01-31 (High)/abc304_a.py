def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    
    # Collect people info: (name, age)
    people = []
    index = 1
    for _ in range(N):
        name = input_data[index]
        age = int(input_data[index + 1])
        people.append((name, age))
        index += 2

    # Find the index of the youngest person (minimum age)
    start_index = min(range(N), key=lambda i: people[i][1])
    
    # Print the names in clockwise order starting from the youngest person
    for i in range(N):
        print(people[(start_index + i) % N][0])

if __name__ == '__main__':
    main()