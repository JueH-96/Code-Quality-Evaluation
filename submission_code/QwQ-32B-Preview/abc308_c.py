import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    people = []
    idx = 1
    for i in range(1, N+1):
        A = int(data[idx])
        B = int(data[idx+1])
        people.append( (i, A, B) )
        idx += 2
    sorted_people = sorted(people, key=lambda x: (-x[1]/(x[1]+x[2]), x[0]))
    result = [str(p[0]) for p in sorted_people]
    print(' '.join(result))

if __name__ == '__main__':
    main()