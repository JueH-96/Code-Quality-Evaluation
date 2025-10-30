def main():
    # Read input
    n = int(input())
    a = list(map(int, input().split()))
    
    # Map each person to the person standing behind them
    person_behind = [0] * (n + 1)  # +1 because person numbers start from 1
    
    # Find the person at the front and prepare the mapping
    front_person = 0
    for i in range(1, n + 1):
        if a[i - 1] == -1:  # Person i is at the front
            front_person = i
        else:  # Person i is right behind person A_i
            person_behind[a[i - 1]] = i
    
    # Traverse the linked list from front to back
    arrangement = [front_person]
    current_person = front_person
    while person_behind[current_person] != 0:
        current_person = person_behind[current_person]
        arrangement.append(current_person)
    
    # Print the result
    print(" ".join(map(str, arrangement)))

if __name__ == "__main__":
    main()