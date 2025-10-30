# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    people = []
    index = 1
    for _ in range(N):
        S = data[index]
        A = int(data[index+1])
        people.append({'name': S, 'age': A})
        index += 2
    
    # Find the youngest person
    youngest = min(people, key=lambda x: x['age'])
    youngest_index = people.index(youngest)
    
    # Reorder the list starting from the youngest
    ordered_people = people[youngest_index:] + people[:youngest_index]
    
    for person in ordered_people:
        print(person['name'])

if __name__ == "__main__":
    main()