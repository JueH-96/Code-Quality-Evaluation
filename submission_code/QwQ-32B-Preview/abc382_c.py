def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    person_idx = 0
    for j in range(M):
        while person_idx < N and A[person_idx] > B[j]:
            person_idx += 1
        if person_idx < N:
            print(person_idx + 1)
        else:
            print(-1)

if __name__ == "__main__":
    main()