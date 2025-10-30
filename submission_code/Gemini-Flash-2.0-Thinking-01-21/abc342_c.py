def solve():
    n = int(input())
    s = input()
    q = int(input())
    operations = []
    for _ in range(q):
        line = input().split()
        operations.append((line[0], line[1]))
    
    current_s = s
    for operation in operations:
        c_i, d_i = operation
        next_s = ""
        for char in current_s:
            if char == c_i:
                next_s += d_i
            else:
                next_s += char
        current_s = next_s
        
    print(current_s)

if __name__ == '__main__':
    solve()