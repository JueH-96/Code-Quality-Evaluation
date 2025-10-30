def solve():
    n = int(input())
    a = list(map(int, input().split()))
    next_person_behind = [0] * (n + 1)
    start_person = -1
    for i in range(1, n + 1):
        if a[i-1] == -1:
            start_person = i
            break
    for i in range(1, n + 1):
        if a[i-1] != -1:
            next_person_behind[a[i-1]] = i
            
    line = []
    current_person = start_person
    while current_person != 0:
        line.append(str(current_person))
        current_person = next_person_behind[current_person]
        
    print(" ".join(line))

if __name__ == '__main__':
    solve()