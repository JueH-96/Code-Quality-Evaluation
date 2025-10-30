def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # This will hold the person who is right behind each person (if any)
    # person_behind[i] = j means person j is right behind person i
    person_behind = [0] * (N + 1)
    
    # Find the start of the line
    start = -1
    for i in range(N):
        if A[i] == -1:
            start = i + 1
        else:
            person_behind[A[i]] = i + 1
    
    # Generate the order of the line starting from the start person
    result = []
    current = start
    while current != 0:
        result.append(current)
        current = person_behind[current]
    
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()