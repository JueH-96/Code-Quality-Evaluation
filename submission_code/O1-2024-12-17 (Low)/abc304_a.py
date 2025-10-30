def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    N = int(input_data[0])
    people = []
    
    idx = 1
    for _ in range(N):
        name = input_data[idx]
        age = int(input_data[idx + 1])
        people.append((name, age))
        idx += 2
    
    # Find the person with the minimum age
    youngest_age = min(people, key=lambda x: x[1])[1]
    
    # Identify the start index (the youngest person)
    start_idx = 0
    for i, (name, age) in enumerate(people):
        if age == youngest_age:
            start_idx = i
            break
    
    # Print names in clockwise order starting from the youngest
    for i in range(N):
        print(people[(start_idx + i) % N][0])

# Do not forget to call the main function
if __name__ == "__main__":
    main()