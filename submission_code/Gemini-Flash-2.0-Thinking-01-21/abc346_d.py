def solve():
    n = int(input())
    s = input()
    c = list(map(int, input().split()))
    min_cost = float('inf')
    
    for i in range(1, n):
        for first_char_int in [0, 1]:
            target_string = []
            if first_char_int == 0:
                for j in range(1, i + 1):
                    target_string.append(str((j - 1) % 2))
                target_string.append(str((i - 1) % 2))
                last_char = str((i - 1) % 2)
                for j in range(i + 2, n + 1):
                    next_char = str(1 - int(last_char))
                    target_string.append(next_char)
                    last_char = next_char
            else: # first_char_int == 1
                for j in range(1, i + 1):
                    target_string.append(str(j % 2))
                target_string.append(str(i % 2))
                last_char = str(i % 2)
                for j in range(i + 2, n + 1):
                    next_char = str(1 - int(last_char))
                    target_string.append(next_char)
                    last_char = next_char
                    
            current_cost = 0
            for j in range(n):
                if s[j] != target_string[j]:
                    current_cost += c[j]
            min_cost = min(min_cost, current_cost)
            
    print(min_cost)

if __name__ == '__main__':
    solve()