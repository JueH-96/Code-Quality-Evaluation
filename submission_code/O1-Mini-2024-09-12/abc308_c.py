import sys

def main():
    import sys
    import sys
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    people = []
    idx = 1
    for i in range(1, N+1):
        A = int(N_and_rest[idx])
        B = int(N_and_rest[idx+1])
        idx +=2
        ratio = A / (A + B)
        people.append( (-ratio, i))
    people.sort()
    result = [str(person[1]) for person in people]
    print(' '.join(result))

if __name__ == "__main__":
    main()