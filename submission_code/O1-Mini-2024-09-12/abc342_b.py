def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx]); idx +=1
    P = list(map(int, data[idx:idx+N])); idx +=N
    Q = int(data[idx]); idx +=1
    queries = [tuple(map(int, data[idx+i*2:idx+i*2+2])) for i in range(Q)]
    
    # Map person number to position
    person_pos = {person: pos for pos, person in enumerate(P, 1)}
    
    for A, B in queries:
        if person_pos[A] < person_pos[B]:
            print(A)
        else:
            print(B)

if __name__ == "__main__":
    main()