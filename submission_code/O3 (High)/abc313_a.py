import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    P = list(map(int, data[1:1 + N]))
    
    # Person 1's current ability
    p1 = P[0]
    
    # If there are no other people, person 1 is already the strongest
    if N == 1:
        print(0)
        return
    
    # Maximum ability among the other people
    strongest_other = max(P[1:])
    
    # Additional points needed so that p1 becomes strictly greater
    if p1 > strongest_other:
        print(0)
    else:
        print(strongest_other - p1 + 1)

if __name__ == "__main__":
    main()