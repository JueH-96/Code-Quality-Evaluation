import sys

def main():
    data = sys.stdin.read().splitlines()
    N = int(data[0])
    people = []
    for i in range(1, N+1):
        name, age = data[i].split()
        people.append( (name, int(age)) )
    
    youngest = min(people, key=lambda x: x[1])
    start_index = people.index(youngest)
    
    new_order = people[start_index:] + people[:start_index]
    
    for person in new_order:
        print(person[0])

if __name__ == "__main__":
    main()