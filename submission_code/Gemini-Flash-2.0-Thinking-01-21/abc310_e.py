def solve():
    n = int(input())
    s = input()
    a = []
    for char in s:
        a.append(int(char))
    
    total_sum = 0
    for i in range(n):
        current_value = a[i]
        total_sum += current_value
        for j in range(i + 1, n):
            current_value = 1 - (current_value * a[j])
            total_sum += current_value
            
    print(total_sum)

if __name__ == '__main__':
    solve()