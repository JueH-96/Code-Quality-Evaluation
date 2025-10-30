def main():
    n = int(input().strip())
    people = []
    ages = []
    
    for _ in range(n):
        data = input().split()
        name = data[0]
        age = int(data[1])
        people.append(name)
        ages.append(age)
        
    min_age = min(ages)
    start_index = ages.index(min_age)
    
    for i in range(n):
        current_index = (start_index + i) % n
        print(people[current_index])

if __name__ == "__main__":
    main()