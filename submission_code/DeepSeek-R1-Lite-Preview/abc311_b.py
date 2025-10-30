def main():
    import sys
    lines = sys.stdin.read().splitlines()
    if not lines:
        print(0)
        return
    N_D = lines[0].split()
    N = int(N_D[0])
    D = int(N_D[1])
    schedules = lines[1:N+1]
    
    all_free = [True] * D
    
    for j in range(D):
        for i in range(N):
            if schedules[i][j] == 'x':
                all_free[j] = False
                break
    
    max_seq = 0
    current_seq = 0
    for j in range(D):
        if all_free[j]:
            current_seq += 1
            if current_seq > max_seq:
                max_seq = current_seq
        else:
            current_seq = 0
    print(max_seq)

if __name__ == "__main__":
    main()