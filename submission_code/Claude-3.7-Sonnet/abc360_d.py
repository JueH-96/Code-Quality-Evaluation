def solve_ants_passing():
    n, t = map(int, input().split())
    s = input()
    positions = list(map(int, input().split()))

    count = 0
    for i in range(n):
        for j in range(i+1, n):
            vi = 1 if s[i] == '1' else -1
            vj = 1 if s[j] == '1' else -1
            
            # If ants are moving with the same velocity, they'll never cross
            if vi == vj:
                continue
            
            time_of_crossing = (positions[j] - positions[i]) / (vi - vj)
            
            if 0 < time_of_crossing <= t + 0.1:
                count += 1

    return count

if __name__ == "__main__":
    print(solve_ants_passing())