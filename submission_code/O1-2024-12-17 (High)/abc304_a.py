def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    # Parse input
    people = []
    idx = 1
    for _ in range(N):
        name = input_data[idx]
        age = int(input_data[idx+1])
        people.append((name, age))
        idx += 2
    
    # Find the index of the youngest person
    youngest_index = min(range(N), key=lambda i: people[i][1])
    
    # Print names starting from the youngest person in clockwise order
    for i in range(N):
        print(people[(youngest_index + i) % N][0])

# Do not forget to call main()        
main()