def solve():
    n, s = map(int, input().split())
    a_initial = list(map(int, input().split()))
    
    infinite_sequence = []
    for i in range(2 * n):
        infinite_sequence.append(a_initial[i % n])
        
    found_sum = False
    for start_index in range(n):
        current_sum = 0
        for length in range(1, 2 * n + 1):
            index = start_index + length - 1
            term = infinite_sequence[index]
            current_sum += term
            if current_sum == s:
                found_sum = True
                break
            if current_sum > s:
                break
        if found_sum:
            break
            
    if found_sum:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()