def solve():
    N_Q = input().split()
    N = int(N_Q[0])
    Q = int(N_Q[1])
    T = list(map(int, input().split()))
    
    teeth = set(range(1, N + 1))
    
    for ti in T:
        if ti in teeth:
            teeth.remove(ti)
        else:
            teeth.add(ti)
    
    print(len(teeth))

if __name__ == "__main__":
    solve()