from math import sqrt

def is_infected(person1, person2, D):
    return sqrt((person1[0] - person2[0])**2 + (person1[1] - person2[1])**2) <= D

def spread_virus(N, D, people):
    infected = [False] * N
    infected[0] = True
    changed = True
    
    while changed:
        changed = False
        for i in range(N):
            if not infected[i]:
                continue
            for j in range(N):
                if not infected[j] and is_infected(people[i], people[j], D):
                    infected[j] = True
                    changed = True
    
    return infected

def main():
    N, D = map(int, input().split())
    people = [tuple(map(int, input().split())) for _ in range(N)]
    
    infected = spread_virus(N, D, people)
    
    for status in infected:
        print("Yes" if status else "No")

if __name__ == "__main__":
    main()