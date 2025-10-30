def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    max_a = a[0]
    for x in a:
        if x > max_a:
            max_a = x
            
    max_b = b[0]
    for x in b:
        if x > max_b:
            max_b = x
            
    result = max_a + max_b
    print(result)

if __name__ == '__main__':
    solve()