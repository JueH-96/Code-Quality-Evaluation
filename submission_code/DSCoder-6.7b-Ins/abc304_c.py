import sys
import math

def read_input():
    N, D = map(int, sys.stdin.readline().split())
    people = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    return N, D, people

def calculate_distance(person1, person2):
    return math.sqrt((person1[0] - person2[0])**2 + (person1[1] - person2[1])**2)

def is_infected(person, others, D):
    for other in others:
        if calculate_distance(person, other) <= D:
            return True
    return False

def solve():
    N, D, people = read_input()
    for i in range(N):
        if is_infected(people[i], people[:i] + people[i+1:], D):
            print("Yes")
        else:
            print("No")

solve()