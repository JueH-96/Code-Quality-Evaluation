def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    people = []
    
    for i in range(1, N + 1):
        name, age = data[i].split()
        age = int(age)
        people.append((name, age))
    
    # Sort people by age to find the youngest
    people.sort(key=lambda x: x[1])
    
    # The youngest person's index in the sorted list is 0
    # We need to print names starting from this person in the original order
    # Since the list is sorted by age, we just print names from the sorted list
    for name, age in people:
        print(name)

if __name__ == "__main__":
    main()