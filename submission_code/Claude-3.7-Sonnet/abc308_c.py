from fractions import Fraction

def solve():
    N = int(input())
    people = []
    
    for i in range(1, N+1):
        A, B = map(int, input().split())
        success_rate = Fraction(A, A + B)
        people.append((i, success_rate))
    
    # Sort people by success rate in descending order
    # If success rates are equal, sort by person number in ascending order
    people.sort(key=lambda x: (-x[1], x[0]))
    
    # Output the sorted list of people
    print(" ".join(map(str, [person[0] for person in people])))

solve()