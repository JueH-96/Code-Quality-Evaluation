def main():
    N = int(input())
    people = []
    for _ in range(N):
        C = int(input())
        A = list(map(int, input().split()))
        people.append((C, A))
    X = int(input())
    
    candidates = []
    for idx in range(N):
        c_i, a_list = people[idx]
        if X in a_list:
            candidates.append((c_i, idx + 1))  # Store (C_i, 1-based index)
    
    if not candidates:
        print(0)
    else:
        min_c = min(c[0] for c in candidates)
        selected = [c[1] for c in candidates if c[0] == min_c]
        selected.sort()
        print(len(selected))
        print(' '.join(map(str, selected)))

if __name__ == "__main__":
    main()