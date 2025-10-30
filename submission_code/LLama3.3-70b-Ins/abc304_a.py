import sys

def solve():
    # Read the number of people
    N = int(input())

    # Read the names and ages of all people
    people = []
    for _ in range(N):
        name, age = input().split()
        people.append((name, int(age)))

    # Find the index of the youngest person
    youngest_index = min(range(N), key=lambda i: people[i][1])

    # Print the names of all people in the order of their seating positions
    for i in range(N):
        print(people[(youngest_index + i) % N][0])

if __name__ == "__main__":
    solve()